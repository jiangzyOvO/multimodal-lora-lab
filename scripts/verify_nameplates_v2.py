"""Validate v2 pixels, independent rendered-text labels, grouping and train arms."""
import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image
from evaluate_nameplates import ROOT, FIELDS, valid_format


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def answer_from_annotations(spec):
    """Read actually rendered value annotations, independently of spec.answer."""
    answer = {}
    for field in FIELDS:
        values = [x["text"] for x in spec["rendered_text"] if x["kind"] == "value" and x["field"] == field]
        require(len(values) <= 1, "Duplicate rendered value")
        if not values or values[0] in ("--", "N/A"):
            answer[field] = None
        elif field == "型号":
            answer[field] = values[0]
        else:
            value = re.sub(r"\s+", "", values[0])
            answer[field] = re.sub(r"AC|DC", "", value) if field == "电压" else value
    return answer


def verify(directory):
    manifest = read(directory / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        require(digest(directory / name) == expected, f"Data fingerprint mismatch: {name}")
    require(digest(ROOT / "scripts/generate_nameplates_v2.py") == manifest["generator_sha256"],
            "Generator fingerprint mismatch")
    require(digest(ROOT / "scripts/generate_nameplates.py") == manifest["renderer_sha256"],
            "Renderer fingerprint mismatch")
    specs = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines() if line]
    by_image = {s["image"]: s for s in specs}
    require(len(by_image) == len(specs) == 1200, "Unexpected image count")
    groups, split_sets = defaultdict(list), defaultdict(lambda: defaultdict(set))
    for spec in specs:
        path = (ROOT / "data" / spec["image"]).resolve()
        require(path.is_relative_to(directory.resolve()), "Image path outside dataset")
        require(digest(path) == spec["image_sha256"], "Image fingerprint mismatch")
        with Image.open(path) as image:
            image.load()
            require(image.size == tuple(manifest["image_size"]), "Image dimensions differ")
        require(valid_format(spec["answer"]), "Invalid answer schema")
        require(answer_from_annotations(spec) == spec["answer"], "Rendered text/answer mismatch")
        for annotation in spec["rendered_text"]:
            x1, y1, x2, y2 = annotation["bbox"]
            require(0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height,
                    "Text outside image")
        groups[(spec["arm"], spec["parent_group"])].append(spec)
        for key in ("parent_group", "model_group", "template", "image_sha256"):
            split_sets[spec["split"]][key].add(spec[key])
        if spec["missing_mode"] == "omit":
            require(not any(x.get("field") == spec["missing_field"] for x in spec["rendered_text"]),
                    "Omitted field is still rendered")
    for left, right in itertools.combinations(split_sets, 2):
        for key in split_sets[left]:
            require(not split_sets[left][key] & split_sets[right][key], f"Split leakage: {key}")
    original_specs = [json.loads(line) for line in (ROOT / "data/nameplates_v1/metadata.jsonl").read_text().splitlines()]
    for key in ("model_group", "image_sha256"):
        old = {row[key] for row in original_specs}
        require(not old & {row[key] for row in specs}, f"Overlap with v1: {key}")
    for (arm, group), rows in groups.items():
        require(len(rows) == 4, f"Expected four variants: {group}")
        require(len({row["image_sha256"] for row in rows}) == 4, f"Duplicate group images: {group}")
        if arm != "control":
            require({r["variant"] for r in rows} == {"complete", "blank", "omit", "marker"}, "Missing pair variant")
            complete = next(r for r in rows if r["variant"] == "complete")
            for row in rows:
                focus = row["focus_field"]
                for field in FIELDS:
                    if field != focus:
                        require(row["display_values"][field] == complete["display_values"][field],
                                "Non-target value changed in pair")
                for key in ("render_seed", "field_order", "field_labels", "font", "template", "distractors"):
                    require(row[key] == complete[key], f"Non-target context changed: {key}")
    old_train = read(ROOT / "data/nameplates_v1/train.json")
    old_by_image = {r["images"][0]: r for r in old_train}
    for key, count in manifest["sizes"].items():
        rows = read(directory / f"{key}.json")
        require(len(rows) == count, f"Wrong record count: {key}")
        require(len({row["images"][0] for row in rows}) == count, "Duplicate image reference")
        image_hashes = []
        for row in rows:
            name = row["images"][0]
            if name in old_by_image:
                require(key in ("train_control", "train_hard") and row == old_by_image[name],
                        "Unexpected or modified v1 record")
            else:
                spec = by_image[name]
                expected_split = "train" if key.startswith("train_") else key
                require(spec["split"] == expected_split, "Record crossed splits")
                if key.startswith("train_"):
                    require(spec["arm"] == key.split("_")[1], "Record crossed train arms")
                require(row["messages"][0] == {"role": "user", "content": manifest["prompt"]}, "Prompt differs")
                require(json.loads(row["messages"][1]["content"]) == spec["answer"], "Record label mismatch")
                image_hashes.append(spec["image_sha256"])
        require(len(image_hashes) == len(set(image_hashes)), f"Duplicate image bytes: {key}")
        if key in ("train_control", "train_hard"):
            require({r["images"][0] for r in rows if r["images"][0] in old_by_image} == set(old_by_image),
                    "Original training records missing")
    for left, right in zip(read(directory / "train_control.json"), read(directory / "train_hard.json")):
        left_old = left["images"][0] in old_by_image
        right_old = right["images"][0] in old_by_image
        require(left_old == right_old and (not left_old or left == right),
                "Shared v1 records have different positions across arms")
    return {"passed": True, "images_checked": len(specs), "sizes": manifest["sizes"],
            "groups": manifest["group_counts"], "cross_split_leakage": 0,
            "label_check": "rendered text annotations; not independent human OCR of all pixels"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/nameplates_v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Report already exists")
    result = verify(args.dataset_dir)
    result["verifier_sha256"] = digest(Path(__file__))
    result["manifest_sha256"] = digest(args.dataset_dir / "manifest.json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
