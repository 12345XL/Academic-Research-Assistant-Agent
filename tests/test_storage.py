"""Offline object-storage checks; metadata and ETags are not integrity proof."""

import hashlib
import io

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from research_agent.settings import ConfigurationError, Settings
from research_agent.storage import S3ObjectStore, StorageError


def configured(**overrides):
    return Settings.from_env({
        "DATABASE_URL": "postgresql://test-user:database-secret@127.0.0.1/test",
        "S3_ACCESS_KEY_ID": "test-access-key",
        "S3_SECRET_ACCESS_KEY": "test-storage-secret",
        **overrides,
    })


def client_error(code, operation):
    return ClientError({"Error": {"Code": code, "Message": "SENSITIVE_SDK_DETAIL"}}, operation)


class FakeS3:
    """Keep bytes independently from metadata, including deliberate corruption."""

    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.reads = []
        self.bodies = []
        self.put_failure = None
        self.corrupt_after_upload = False

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise client_error("NoSuchKey", "HeadObject")
        item = self.objects[Key]
        return {"ContentLength": len(item["body"]), "Metadata": item["metadata"],
                "ETag": '"untrusted-etag"', "ContentType": "application/json"}

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise client_error("NoSuchKey", "GetObject")
        self.reads.append(Key)
        body = io.BytesIO(self.objects[Key]["body"])
        self.bodies.append(body)
        return {"Body": body}

    def put_object(self, *, Bucket, Key, Body, ContentType, Metadata):
        if self.put_failure:
            raise self.put_failure
        self.uploads.append(Key)
        content = bytes(Body)
        if self.corrupt_after_upload:
            content = b"!" + content[1:]
        self.objects[Key] = {"body": content, "metadata": dict(Metadata)}
        return {"ETag": '"plausible-but-not-a-sha256"'}


def test_put_reads_actual_bytes_and_second_upload_is_idempotent():
    client = FakeS3()
    store = S3ObjectStore(configured(), client)
    content = b'{"paper":"evidence"}'
    checksum = hashlib.sha256(content).hexdigest()
    assert store.put_verified("paper.json", content, checksum) is True
    assert store.put_verified("paper.json", content, checksum) is False
    assert client.uploads == ["paper.json"]
    assert client.reads == ["paper.json", "paper.json"]
    assert client.objects["paper.json"]["body"] == content
    assert all(body.closed for body in client.bodies)


def test_forged_metadata_with_same_length_corrupt_bytes_is_repaired():
    client = FakeS3()
    content = b'{"value":"real"}'
    checksum = hashlib.sha256(content).hexdigest()
    client.objects["paper.json"] = {
        "body": b'{"value":"fake"}', "metadata": {"sha256": checksum},
    }
    assert S3ObjectStore(configured(), client).put_verified("paper.json", content, checksum) is True
    assert client.objects["paper.json"]["body"] == content
    assert client.reads == ["paper.json", "paper.json"]


def test_good_bytes_do_not_require_trusting_or_rewriting_metadata():
    client = FakeS3()
    content = b"original evidence"
    checksum = hashlib.sha256(content).hexdigest()
    client.objects["paper.json"] = {"body": content, "metadata": {"sha256": "WRONG"}}
    assert S3ObjectStore(configured(), client).put_verified("paper.json", content, checksum) is False
    assert client.uploads == []
    assert client.reads == ["paper.json"]


def test_invalid_input_checksum_rejected_before_storage_io():
    client = FakeS3()
    with pytest.raises(StorageError, match="does not match"):
        S3ObjectStore(configured(), client).put_verified("paper.json", b"actual bytes", "0" * 64)
    assert not client.objects and not client.uploads and not client.reads


def test_upload_success_response_with_corrupt_readback_is_failure():
    client = FakeS3()
    client.corrupt_after_upload = True
    content = b'{"paper":"real"}'
    with pytest.raises(StorageError, match="read-back checksum"):
        S3ObjectStore(configured(), client).put_verified("paper.json", content, hashlib.sha256(content).hexdigest())
    assert client.uploads == ["paper.json"]
    assert all(body.closed for body in client.bodies)


@pytest.mark.parametrize("failure", [
    client_error("AccessDenied", "PutObject"),
    EndpointConnectionError(endpoint_url="http://SENSITIVE_SDK_DETAIL"),
])
def test_upload_failure_has_safe_public_error(failure):
    client = FakeS3()
    client.put_failure = failure
    content = b"document"
    with pytest.raises(StorageError, match="could not be stored") as caught:
        S3ObjectStore(configured(), client).put_verified("paper.json", content, hashlib.sha256(content).hexdigest())
    assert "SENSITIVE_SDK_DETAIL" not in str(caught.value)
    assert client.objects == {}


def test_missing_object_is_distinct_from_access_denied(monkeypatch):
    client = FakeS3()
    store = S3ObjectStore(configured(), client)
    assert store.head("missing.json") is None
    with pytest.raises(StorageError, match="could not be read"):
        store.read_bytes("missing.json")

    def deny(**kwargs):
        raise client_error("AccessDenied", "HeadObject")

    monkeypatch.setattr(client, "head_object", deny)
    with pytest.raises(StorageError, match="metadata could not be read") as caught:
        store.head("private.json")
    assert "SENSITIVE_SDK_DETAIL" not in str(caught.value)


def test_settings_do_not_expose_credentials_in_repr_or_validation_errors():
    settings = configured()
    representation = repr(settings)
    for secret in ("test-user", "database-secret", "test-access-key", "test-storage-secret"):
        assert secret not in representation
    assert "research-papers" in representation
    with pytest.raises(ConfigurationError) as caught:
        configured(S3_ENDPOINT_URL="http://user:SENSITIVE_SECRET@127.0.0.1:8333")
    assert "SENSITIVE_SECRET" not in str(caught.value)
    with pytest.raises(ConfigurationError, match="Missing configuration"):
        Settings.from_env({})


def test_bucket_access_denied_never_attempts_creation():
    class DeniedBucket:
        def head_bucket(self, **kwargs):
            raise client_error("AccessDenied", "HeadBucket")

        def create_bucket(self, **kwargs):
            pytest.fail("Access denied must not be treated as a missing bucket")

    store = S3ObjectStore(configured(), DeniedBucket())
    with pytest.raises(StorageError, match="cannot be accessed"):
        store.ensure_bucket()
    assert store.health() == {"status": "unavailable", "bucket": "research-papers"}
