"""Generate a grouped synthetic stress set and equal-size control/hard train arms.

Both arms contain the same 400 original training images plus 400 new images.
The intervention is the complete data recipe (paired missing-field examples),
not a claim that any one augmentation independently causes an improvement.
"""
import argparse
import copy
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

import PIL
import generate_nameplates as v1

ROOT = v1.ROOT
SEED = 20261010
EXTRA_FONTS = {
    "val": "/System/Library/Fonts/STHeiti Medium.ttc",
    "test": "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
}
NEW_TEMPLATES = {
    "V201": ("val", "splitright", [(44, 100, 344, 206), (44, 228, 344, 338), (366, 100, 628, 338)]),
    "V202": ("val", "steprows", [(44, 98, 580, 166), (68, 184, 604, 252), (92, 270, 628, 338)]),
    "V203": ("val", "widebottom", [(44, 98, 290, 206), (312, 98, 628, 206), (44, 232, 628, 338)]),
    "V204": ("val", "threecolumns", [(44, 98, 220, 336), (242, 98, 436, 336), (458, 98, 628, 336)]),
    "E201": ("test", "splitleft", [(44, 98, 286, 338), (308, 98, 628, 198), (308, 220, 628, 338)]),
    "E202": ("test", "reversesteps", [(92, 98, 628, 166), (68, 184, 604, 252), (44, 270, 580, 338)]),
    "E203": ("test", "widetop", [(44, 98, 628, 196), (44, 220, 356, 338), (378, 220, 628, 338)]),
    "E204": ("test", "staggeredcolumns", [(44, 120, 238, 338), (260, 98, 442, 316), (464, 110, 628, 328)]),
}
NOVEL_VALUES = {
    "val": ([9, 18, 72, 120, 240, 400], [35, 90, 160, 280, 650, 1600]),
    "test": ([6, 15, 30, 60, 127, 208, 277, 450], [7, 45, 85, 125, 420, 850, 2500, 4200]),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def base_spec(rng, split, index):
    templates = [name for name, (s, _, _) in v1.TEMPLATES.items() if s == "train"] if split == "train" else [
        name for name, (s, _, _) in NEW_TEMPLATES.items() if s == split]
    # Cross font and value novelty independently within each evaluation split.
    novel_values = split != "train" and index % 2 == 1
    novel_font = split != "train" and (index // 2) % 2 == 1
    voltages, powers = NOVEL_VALUES[split] if novel_values else (v1.VOLTAGES, v1.POWERS)
    voltage, power = rng.choice(voltages), rng.choice(powers)
    group = f"v2-{split}-{index:03d}"
    model = f"{rng.choice(['AB', 'CX', 'DK', 'KR', 'QS', 'SV'])}-{dict(train=1000, val=2000, test=3000)[split] + index}"
    order = list(itertools.permutations(v1.FIELDS))[index % 6]
    return {
        "split": split, "parent_group": group, "model_group": model,
        "template": templates[(index // 4) % len(templates)],
        "font": EXTRA_FONTS[split] if novel_font else v1.FONTS[index % len(v1.FONTS)],
        "font_novelty": "unseen" if novel_font else "seen",
        "value_novelty": "unseen" if novel_values else "seen",
        "render_seed": rng.getrandbits(32), "field_order": list(order),
        "field_labels": {f: rng.choice(v1.LABELS[f]) for f in v1.FIELDS},
        "answer": {"型号": model, "电压": f"{voltage}V", "功率": f"{power}W"},
        "display_values": {"型号": model, "电压": f"{'AC' if voltage >= 110 else 'DC'} {voltage} V",
                           "功率": f"{power} W"},
        "missing_field": None, "missing_mode": None,
        "distractors": [f"S/N: {rng.randrange(10000000, 99999999)}    Frequency: {rng.choice([50, 60])} Hz",
                        f"Date: 2025-{rng.randrange(1, 13):02d}    Weight: {rng.randrange(1, 30)} kg"],
    }


def missing_variant(base, field, mode, marker="--"):
    spec = copy.deepcopy(base)
    spec["missing_field"], spec["missing_mode"] = field, mode
    if field:
        spec["answer"][field] = None
        spec["display_values"][field] = marker if mode == "marker" else ""
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/nameplates_v2")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error("Dataset directory already exists; use a new version")
    if not destination.is_relative_to(ROOT / "data"):
        parser.error("Dataset must be under the repository data directory")
    for font in [*v1.FONTS, *EXTRA_FONTS.values()]:
        if not Path(font).is_file():
            parser.error(f"Missing required font: {font}")
    v1.TEMPLATES.update(NEW_TEMPLATES)
    destination.mkdir(parents=True)
    (destination / "images").mkdir()
    metadata, datasets = [], {key: [] for key in ("train_control_extra", "train_hard_extra", "val", "test")}
    rng = random.Random(SEED)

    def emit(spec, arm, suffix):
        spec["arm"] = arm
        spec["id"] = f"{spec['parent_group']}-{arm}-{suffix}"
        path = destination / "images" / f"{spec['id']}.png"
        spec["rendered_text"] = v1.render(spec, path)
        spec["image"] = path.relative_to(ROOT / "data").as_posix()
        spec["image_sha256"] = digest(path)
        metadata.append(spec)
        key = f"train_{arm}_extra" if spec["split"] == "train" else spec["split"]
        datasets[key].append({"messages": [
            {"role": "user", "content": v1.PROMPT},
            {"role": "assistant", "content": json.dumps(spec["answer"], ensure_ascii=False, separators=(",", ":"))},
        ], "images": [spec["image"]]})

    # 100 shared parent groups in each train arm; 50 groups per eval split.
    control_missing = [None] * 180 + [f for f in v1.FIELDS for _ in range(40)]
    rng.shuffle(control_missing)
    for split, count in (("train", 100), ("val", 50), ("test", 50)):
        for index in range(count):
            base = base_spec(rng, split, index)
            focus = v1.FIELDS[index % 3]
            arm = "hard" if split == "train" else "evaluation"
            for variant, mode in enumerate((None, "blank", "omit", "marker")):
                spec = missing_variant(base, focus if mode else None, mode,
                                       marker="--" if index % 2 == 0 else "N/A")
                spec["variant"] = "complete" if mode is None else mode
                spec["focus_field"] = focus
                emit(spec, arm, str(variant))
            if split == "train":
                for variant in range(4):
                    control = copy.deepcopy(base)
                    # Same group count, identities, fonts and templates. Control
                    # varies layout order; hard arm fixes context within a pair.
                    permutations = list(itertools.permutations(v1.FIELDS))
                    start = permutations.index(tuple(base["field_order"]))
                    control["field_order"] = list(permutations[(start + variant) % 6])
                    field = None if variant == 0 else control_missing[index * 3 + variant - 1]
                    mode = rng.choice(["blank", "omit", "marker"]) if field else None
                    control = missing_variant(control, field, mode, rng.choice(["--", "N/A"]))
                    control["variant"] = f"ordinary_{variant}"
                    control["focus_field"] = None
                    emit(control, "control", str(variant))
    original_train = json.loads((ROOT / "data/nameplates_v1/train.json").read_text())
    for arm in ("control", "hard"):
        datasets[f"train_{arm}"] = original_train + datasets[f"train_{arm}_extra"]
    for key, rows in datasets.items():
        # Keep the 400 shared v1 records at identical positions across train
        # arms so their ordering is not an additional experimental difference.
        shuffle_seed = SEED if key in ("train_control", "train_hard") else SEED + sum(map(ord, key))
        random.Random(shuffle_seed).shuffle(rows)
        dump(destination / f"{key}.json", rows)
    (destination / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata), encoding="utf-8")
    registry = {}
    for key in ("train_control", "train_hard", "val", "test"):
        registry[f"nameplate_v2_{key}"] = {
            "file_name": f"{key}.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    dump(destination / "dataset_info.json", registry)
    manifest = {
        "version": 2, "seed": SEED, "kind": "synthetic_counterfactual_stress_test",
        "image_size": v1.SIZE, "prompt": v1.PROMPT, "pillow": PIL.__version__,
        "sizes": {key: len(rows) for key, rows in datasets.items()},
        "group_counts": {split: len({r['parent_group'] for r in metadata if r['split'] == split})
                         for split in ("train", "val", "test")},
        "recipe": "100 train parent groups per arm; hard groups have complete/blank/omit/marker variants. "
                  "Control retains 70% complete images and varies order within groups. "
                  "Compare the whole recipe, not an isolated augmentation effect.",
        "limitations": ["All images are synthetic and share the v1 drawing engine.",
                        "200 evaluation images represent 50 correlated groups, not 200 independent devices.",
                        "Only integer V/W labels are supported; values are fictional.",
                        "Final test predictions must wait until checkpoint selection is frozen."],
        "generator_sha256": digest(Path(__file__)), "renderer_sha256": digest(Path(v1.__file__)),
        "original_train_sha256": digest(ROOT / "data/nameplates_v1/train.json"),
        "font_sha256": {font: digest(Path(font)) for font in [*v1.FONTS, *EXTRA_FONTS.values()]},
        "files_sha256": {path.name: digest(path) for path in sorted(destination.glob("*.json*"))},
        "missing_distribution": {
            arm: dict(Counter(f"{row['missing_field'] or 'none'}:{row['missing_mode'] or 'none'}"
                              for row in metadata if row['arm'] == arm))
            for arm in ("control", "hard", "evaluation")},
    }
    dump(destination / "manifest.json", manifest)
    print(json.dumps({"sizes": manifest["sizes"], "groups": manifest["group_counts"]}, indent=2))


if __name__ == "__main__":
    main()
