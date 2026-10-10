"""Fetch official demo at a fixed commit; split by image, not by dialogue."""
import hashlib
import json
import urllib.request
from pathlib import Path

from PIL import Image

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
revision = "ce9dc9e072f80fa3abe0989d4ab90da25f083438"
files = ["mllm_demo.json"] + [f"mllm_demo_data/{i}.jpg" for i in (1, 2, 3)]
manifest = {"source_revision": revision, "files": {}}
for name in files:
    path = data_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/{revision}/data/{name}"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    manifest["files"][name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(name, path.stat().st_size, flush=True)

records = json.loads((data_dir / "mllm_demo.json").read_text())
train, holdout = [], []
for record in records:
    assert sum(m["content"].count("<image>") for m in record["messages"]) == len(record["images"])
    assert all(m["content"].strip() for m in record["messages"])
    for name in record["images"]:
        with Image.open(data_dir / name) as image:
            image.verify()
    # All dialogues about the astronaut image stay outside the training set.
    (holdout if "mllm_demo_data/3.jpg" in record["images"] else train).append(record)
train_images = {x for record in train for x in record["images"]}
test_images = {x for record in holdout for x in record["images"]}
assert not train_images & test_images
registry_path = data_dir / "dataset_info.json"
metadata = json.loads(registry_path.read_text()) if registry_path.exists() else {}
for name, rows in [("demo_train", train), ("demo_holdout", holdout)]:
    (data_dir / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    metadata[name] = {"file_name": f"{name}.json", "formatting": "sharegpt",
                      "columns": {"messages": "messages", "images": "images"},
                      "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
(data_dir / "dataset_info.json").write_text(json.dumps(metadata, indent=2))
manifest.update(train_samples=len(train), holdout_samples=len(holdout), train_images=sorted(train_images), holdout_images=sorted(test_images))
(root / "logs/data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps(manifest, ensure_ascii=False, indent=2))
