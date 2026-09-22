"""Explicit download of the fixed cross-encoder; verify checked-in checksums."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_agent.reranking import MODEL_ID, MODEL_REVISION, model_directory


def main():
    from huggingface_hub import snapshot_download
    manifest = json.loads((ROOT / "src/research_agent/model_manifests/minilm-reranker.json").read_text())
    assert manifest["model"] == MODEL_ID and manifest["revision"] == MODEL_REVISION
    directory = model_directory()
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=directory,
                      allow_patterns=list(manifest["sha256"]), max_workers=4)
    for name, checksum in manifest["sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Downloaded reranker file does not match frozen checksum")
    print(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION, "checksums_verified": True}))


if __name__ == "__main__":
    main()
