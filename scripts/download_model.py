"""Download one pinned upstream model snapshot, including its license."""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
repo = "Qwen/Qwen2-VL-2B-Instruct"
revision = "895c3a49bc3fa70a340399125c650a463535e71c"
destination = root / "models/Qwen2-VL-2B-Instruct"
print(f"Downloading {repo}@{revision} to {destination}", flush=True)
snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination), max_workers=2,
                  allow_patterns=["*.json", "*.safetensors", "merges.txt", "LICENSE", "README.md"])
manifest = {"repo": repo, "revision": revision, "path": str(destination),
            "files": {p.name: p.stat().st_size for p in destination.iterdir() if p.is_file()}}
(root / "logs/model_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
