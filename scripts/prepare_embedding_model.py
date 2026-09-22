"""Download a fixed public model revision; only this explicit script uses HF network."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.embeddings import MODEL_FILES, MODEL_ID, MODEL_REVISION, MODEL_SHA256, model_directory


def main():
    from huggingface_hub import snapshot_download
    directory = model_directory()
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=directory,
                      allow_patterns=MODEL_FILES, max_workers=4)
    manifest = {"model": MODEL_ID, "revision": MODEL_REVISION, "sha256": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in MODEL_FILES}}
    if manifest["sha256"] != MODEL_SHA256:
        raise ValueError("Downloaded model files do not match the frozen checksums")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
