"""Server-owned grants and scoped corpus counts; no provider calls are needed.

RUN_STORAGE_INTEGRATION=1 enables the isolated local PostgreSQL check below.
"""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
import uuid

from dotenv import dotenv_values
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
import pytest

from research_agent.access import AccessError, AccessPolicy
from research_agent.settings import Settings
from research_agent.storage import Repository


TOKEN = "test-only-alice-token"
DIGEST = hashlib.sha256(TOKEN.encode()).hexdigest()


def write_policy(path, *, papers=None, identity="alice", tokens=None):
    records = {DIGEST: {"id": identity, "allowed_paper_ids": ["p1"] if papers is None else papers}}
    path.write_text(json.dumps({"tokens": records if tokens is None else tokens}), encoding="utf-8")


@pytest.fixture
def configured_policy(tmp_path):
    path = tmp_path / "access-policy.json"
    write_policy(path)
    return path, AccessPolicy.from_env({"RESEARCH_ACCESS_POLICY_FILE": str(path)})


def assert_error(call, code, status):
    with pytest.raises(AccessError) as caught:
        call()
    assert caught.value.code == code
    assert caught.value.status_code == status
    assert TOKEN not in str(caught.value) and DIGEST not in str(caught.value)


def test_absent_configuration_keeps_explicit_shared_local_mode():
    policy = AccessPolicy.from_env({})
    principal = policy.authenticate(None)
    assert policy.mode == principal.mode == "local_public"
    assert principal.allowed_paper_ids is None
    assert principal.allows("any-paper")
    policy.require_paper(principal, "any-paper")
    assert policy.revalidate(principal, "any-paper", None) == principal
    assert principal.public_status() == {
        "principal_id": "local-user", "mode": "local_public", "allowed_paper_ids": ["*"],
    }


@pytest.mark.parametrize("filename", ["", "  "])
def test_configured_empty_path_never_disables_authentication(filename):
    assert_error(lambda: AccessPolicy.from_env({"RESEARCH_ACCESS_POLICY_FILE": filename}),
                 "access_policy_unavailable", 503)


def test_authentication_scope_and_public_identity_contain_no_credentials(configured_policy):
    _, policy = configured_policy
    principal = policy.authenticate("bEaReR " + TOKEN)
    assert principal.principal_id == "alice" and principal.mode == "bearer_policy"
    assert principal.allowed_paper_ids == frozenset({"p1"})
    policy.require_paper(principal, "p1")
    assert_error(lambda: policy.require_paper(principal, "p2"), "paper_access_denied", 403)
    assert principal.public_status()["allowed_paper_ids"] == ["p1"]
    for public in (repr(principal), str(asdict(principal)), json.dumps(principal.public_status())):
        assert TOKEN not in public and DIGEST not in public


@pytest.mark.parametrize("header", [
    None, "", TOKEN, "Basic " + TOKEN, "Bearer wrong-token", "Bearer ",
    "Bearer  " + TOKEN, " Bearer " + TOKEN, "Bearer\t" + TOKEN,
    "Bearer " + TOKEN + "\n", "Bearer 非ASCII", "Bearer " + "x" * 8192,
])
def test_invalid_or_missing_credential_is_unauthorized(configured_policy, header):
    _, policy = configured_policy
    assert_error(lambda: policy.authenticate(header), "authentication_required", 401)


@pytest.mark.parametrize("papers,allowed", [([], False), (["*"], True)])
def test_empty_scope_denies_and_only_explicit_wildcard_grants_all(configured_policy, papers, allowed):
    path, policy = configured_policy
    write_policy(path, papers=papers)
    principal = policy.authenticate("Bearer " + TOKEN)
    assert principal.allows("p1") is allowed
    assert principal.allows("not-in-corpus") is allowed


@pytest.mark.parametrize("document", [
    None, [], {}, {"tokens": []}, {"tokens": {}, "other": True},
    {"tokens": {"not-a-digest": {"id": "alice", "allowed_paper_ids": ["p1"]}}},
    {"tokens": {DIGEST: {"id": "", "allowed_paper_ids": []}}},
    {"tokens": {DIGEST: {"id": "alice", "allowed_paper_ids": "*"}}},
    {"tokens": {DIGEST: {"id": "alice", "allowed_paper_ids": ["*", "p1"]}}},
    {"tokens": {DIGEST: {"id": "alice", "allowed_paper_ids": ["p1", "p1"]}}},
    {"tokens": {DIGEST: {"id": "alice", "allowed_paper_ids": [None]}}},
    {"tokens": {DIGEST: {"id": "alice", "allowed_paper_ids": [" "]}}},
    {"tokens": {DIGEST: {"id": "alice", "allowed_paper_ids": ["p1\n"]}}},
    {"tokens": {DIGEST: {"id": "alice", "allowed_paper_ids": [], "admin": True}}},
])
def test_invalid_policy_fails_closed_with_safe_error(configured_policy, document):
    path, policy = configured_policy
    path.write_text(json.dumps(document), encoding="utf-8")
    assert_error(lambda: policy.authenticate("Bearer " + TOKEN), "access_policy_unavailable", 503)


@pytest.mark.parametrize("content", [
    b'{"tokens": {}, "tokens": {}}', b'{"tokens": ', b'\xff', b" " * 1_048_577,
])
def test_unreadable_json_duplicate_keys_and_oversize_fail_closed(configured_policy, content):
    path, policy = configured_policy
    path.write_bytes(content)
    assert_error(lambda: policy.authenticate(None), "access_policy_unavailable", 503)


def test_revalidation_observes_scope_revocation_identity_change_and_token_removal(configured_policy):
    path, policy = configured_policy
    principal = policy.authenticate("Bearer " + TOKEN)
    write_policy(path, papers=["p2"])
    assert principal.allows("p1")  # A cached request snapshot is intentionally insufficient.
    assert_error(lambda: policy.revalidate(principal, "p1", "Bearer " + TOKEN), "paper_access_denied", 403)
    refreshed = policy.revalidate(principal, "p2", "Bearer " + TOKEN)
    assert refreshed.allowed_paper_ids == frozenset({"p2"})
    write_policy(path, identity="different-user")
    assert_error(lambda: policy.revalidate(principal, "p1", "Bearer " + TOKEN), "authentication_required", 401)
    write_policy(path, tokens={})
    assert_error(lambda: policy.revalidate(principal, "p1", "Bearer " + TOKEN), "authentication_required", 401)


def test_deleted_or_broken_policy_does_not_reuse_cached_grants(configured_policy):
    path, policy = configured_policy
    principal = policy.authenticate("Bearer " + TOKEN)
    path.unlink()
    assert_error(lambda: policy.revalidate(principal, "p1", "Bearer " + TOKEN), "access_policy_unavailable", 503)
    path.mkdir()
    assert_error(lambda: policy.authenticate("Bearer " + TOKEN), "access_policy_unavailable", 503)


@pytest.fixture
def scoped_repository():
    """An exact random database; never mutate the live corpus or use object storage."""
    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    base = Settings.from_env(values)
    assert urlparse(base.database_url).hostname == "127.0.0.1"
    database = "research_access_test_" + uuid.uuid4().hex
    admin_url = base.database_url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    repo = Repository(replace(base, database_url=base.database_url.rsplit("/", 1)[0] + "/" + database))
    try:
        repo.migrate()
        with repo.connect() as conn:
            for paper_id, title, current, paragraph_count in [
                ("p1", "Alpha methods", True, 2), ("p2", "Beta methods", True, 1),
                ("p3", "Retired methods", False, 4),
            ]:
                version = uuid.uuid4()
                conn.execute("INSERT INTO stored_objects(object_key,bucket,sha256,size_bytes,content_type,state) "
                             "VALUES (%s,'test',%s,1,'application/json','published')", (paper_id, "0" * 64))
                conn.execute("INSERT INTO papers(paper_id,in_current_corpus) VALUES (%s,%s)", (paper_id, current))
                conn.execute(
                    "INSERT INTO paper_versions(version_id,paper_id,dataset_version,source,split,title,abstract,"
                    "object_key,content_sha256,state,paper_payload) VALUES (%s,%s,'test','qasper','train',%s,'',"
                    "%s,%s,'active',%s)",
                    (version, paper_id, title, paper_id, "0" * 64, Jsonb({"paper_id": paper_id, "title": title})),
                )
                conn.execute("UPDATE papers SET current_version_id=%s WHERE paper_id=%s", (version, paper_id))
                for ordinal in range(paragraph_count):
                    conn.execute(
                        "INSERT INTO paragraphs(version_id,chunk_id,ordinal,section_index,paragraph_index,"
                        "section_name,text,text_sha256,paragraph_payload) VALUES (%s,%s,%s,0,%s,'Results',"
                        "'test',%s,'{}')", (version, f"{paper_id}:{ordinal}", ordinal, ordinal, "0" * 64),
                    )
                conn.execute(
                    "INSERT INTO paper_metadata(paper_id,arxiv_submitted_at,ccf_venue,ccf_level,ccf_catalog_url,"
                    "metadata_source,metadata_checked_at,research_direction) VALUES (%s,now(),'EMNLP','B',"
                    "'https://example.test','test',now(),'nlp')", (paper_id,),
                )
            # Unpublished objects and a prior version must not inflate a restricted summary.
            conn.execute("INSERT INTO stored_objects(object_key,bucket,sha256,size_bytes,content_type,state) "
                         "VALUES ('orphan','test',%s,1,'application/json','orphaned'),"
                         "('old-p1','test',%s,1,'application/json','published')", ("0" * 64, "1" * 64))
            conn.execute(
                "INSERT INTO paper_versions(version_id,paper_id,dataset_version,source,split,title,abstract,"
                "object_key,content_sha256,state,paper_payload) VALUES (%s,'p1','old','qasper','train','Old','',"
                "'old-p1',%s,'archived','{}')", (uuid.uuid4(), "1" * 64),
            )
            conn.execute("UPDATE corpus_state SET revision=7,manifest_sha256=%s,paper_count=2,paragraph_count=3,"
                         "published_at=now() WHERE singleton", ("a" * 64,))
        yield repo
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


@pytest.mark.skipif(os.getenv("RUN_STORAGE_INTEGRATION") != "1", reason="requires local PostgreSQL")
def test_real_sql_scope_limits_rows_totals_and_all_summary_counts(scoped_repository):
    repo = scoped_repository
    public = repo.list_papers()
    assert public["total"] == 2
    scoped = repo.list_papers(q="methods", direction="nlp", allowed_paper_ids=["p1", "p3"], limit=1)
    assert scoped["total"] == 1 and [p["paper_id"] for p in scoped["items"]] == ["p1"]
    empty_page = repo.list_papers(allowed_paper_ids=["p1"], limit=1, offset=1)
    assert empty_page["total"] == 1 and empty_page["items"] == []
    assert repo.list_papers(q="Beta", allowed_paper_ids=["p1"])["total"] == 0
    for scope in ([], ["unknown"], ["p3"], ["p1') OR TRUE --"]):
        listing = repo.list_papers(allowed_paper_ids=scope)
        assert listing["total"] == 0 and listing["items"] == []
        summary = repo.summary(allowed_paper_ids=scope)
        assert summary["papers"] == summary["paper_count"] == 0
        assert summary["paragraphs"] == summary["paragraph_count"] == 0
        assert summary["objects"] == summary["metadata_dates"] == summary["metadata_ccf"] == 0
        assert summary["metadata_categorized"] == 0 and summary["object_states"] == {}
        assert summary["manifest_sha256"] is None
    summary = repo.summary(allowed_paper_ids=["p1", "p1", "p3"])
    assert summary["revision"] == 7 and summary["published_at"]
    assert summary["papers"] == summary["paper_count"] == 1
    assert summary["paragraphs"] == summary["paragraph_count"] == 2
    assert summary["objects"] == summary["metadata_dates"] == summary["metadata_ccf"] == 1
    assert summary["metadata_categorized"] == 1 and summary["object_states"] == {"published": 1}
    assert summary["manifest_sha256"] is None
    # Default call retains the historical all-corpus/admin summary contract.
    unscoped = repo.summary()
    assert unscoped["papers"] == 2 and unscoped["paragraphs"] == 3
    assert unscoped["object_states"] == {"published": 4, "orphaned": 1}
    assert unscoped["metadata_dates"] == unscoped["metadata_ccf"] == unscoped["metadata_categorized"] == 3
    assert unscoped["manifest_sha256"] == "a" * 64
