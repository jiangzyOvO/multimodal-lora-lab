"""Compare saved base/LoRA predictions without model loading or changing v1 scores.

Example:
  .venv/bin/python scripts/analyze_nameplate_predictions.py --split test \
    --output results/analysis_v2/test.json

Normalization is an exploratory, post-hoc diagnostic. It must be frozen before
evaluation on a new holdout; the original strict metric remains unchanged.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from evaluate_nameplates import FIELDS, parse_object, score_response, valid_format

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    "id": "nameplate_normalization_v1",
    "status": "post_hoc_exploratory_on_nameplates_v1",
    "rules": [
        "Remove at most one code fence enclosing the entire response.",
        "Parse the entire JSON object; reject duplicate keys and NaN/Infinity.",
        "Map string null (case-insensitive), --, N/A and whitespace-only strings to null.",
        "Remove whitespace from voltage and power strings; remove AC/DC from voltage.",
        "Preserve model identifiers verbatim except the explicit null markers above.",
        "Do not repair numbers, append units, fill missing keys or drop extra keys.",
    ],
    "limitations": [
        "This diagnostic does not replace strict raw-output accuracy.",
        "The v1 test set has been inspected; future tuning needs a new final holdout.",
        "Normalization accuracy is not an OCR-only metric.",
        "Timing comes from historical runs, not a controlled performance benchmark.",
    ],
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic_object(raw):
    parsed = parse_object(raw)
    if parsed is None:
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), re.S | re.I)
        if match:
            parsed = parse_object(match.group(1))
    return parsed


def normalize_object(value):
    if value is None:
        return None
    normalized = dict(value)
    for field in FIELDS:
        item = normalized.get(field)
        if not isinstance(item, str):
            continue
        if item.strip().upper() in {"NULL", "--", "N/A", ""}:
            normalized[field] = None
        elif field in ("电压", "功率"):
            item = re.sub(r"\s+", "", item)
            if field == "电压":
                item = re.sub(r"AC|DC", "", item, flags=re.I)
            normalized[field] = item
    return normalized


def score_row(row):
    diagnostic = diagnostic_object(row["raw_response"])
    normalized = normalize_object(diagnostic)
    return {
        **row,
        "scores": score_response(row["raw_response"], row["expected"]),
        "diagnostic_object": diagnostic,
        "normalized_object": normalized,
        "normalized_exact_match": normalized == row["expected"],
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def null_metrics(rows, object_key):
    counts = dict(tp=0, fp=0, fn=0, target_null_count=0, target_present_count=0,
                  spurious_value_count=0, unparseable_or_missing_key_count=0)
    for row in rows:
        obj = row[object_key]
        for field in FIELDS:
            available = obj is not None and field in obj
            predicted_null = available and obj[field] is None
            target_null = row["expected"][field] is None
            counts["target_null_count" if target_null else "target_present_count"] += 1
            counts["unparseable_or_missing_key_count"] += not available
            counts["tp"] += target_null and predicted_null
            counts["fp"] += not target_null and predicted_null
            counts["fn"] += target_null and not predicted_null
            counts["spurious_value_count"] += target_null and available and not predicted_null
    return {
        **counts,
        "precision": ratio(counts["tp"], counts["tp"] + counts["fp"]),
        "recall": ratio(counts["tp"], counts["target_null_count"]),
        "spurious_value_rate": ratio(counts["spurious_value_count"], counts["target_null_count"]),
        "false_null_rate": ratio(counts["fp"], counts["target_present_count"]),
    }


def summarize(rows):
    require(bool(rows), "Cannot summarize an empty group")
    n = len(rows)
    counts = {
        "strict_format": sum(row["scores"]["strict_format"] for row in rows),
        "strict_exact_match": sum(row["scores"]["strict_exact_match"] for row in rows),
        "fence_only_exact_match": sum(row["scores"]["diagnostic_exact_match"] for row in rows),
        "normalized_exact_match": sum(row["normalized_exact_match"] for row in rows),
    }
    return {
        "sample_count": n,
        "counts": counts,
        "rates": {key: value / n for key, value in counts.items()},
        "normalized_field_accuracy": {
            field: sum(row["normalized_object"] is not None
                       and field in row["normalized_object"]
                       and row["normalized_object"][field] == row["expected"][field]
                       for row in rows) / n for field in FIELDS
        },
        "null_metrics_fence_only": null_metrics(rows, "diagnostic_object"),
        "null_metrics_normalized": null_metrics(rows, "normalized_object"),
    }


def slice_labels(spec):
    field, mode = spec["missing_field"], spec["missing_mode"]
    return {
        "completeness": "complete" if field is None else "missing",
        "template": spec["template"],
        "font": Path(spec["font"]).name,
        "missing_field": field or "none",
        "missing_mode": mode or "none",
        "missing_field_and_mode": f"{field or 'none'}:{mode or 'none'}",
    }


def load_run(folder, dataset_path):
    run = read_json(folder / "run.json")
    rows = read_jsonl(folder / "predictions.jsonl")
    dataset = read_json(dataset_path)
    require(run["dataset_sha256"] == digest(dataset_path), "Dataset fingerprint mismatch")
    require(len(rows) == len(dataset) == run["sample_count"] > 0,
            "A complete, nonempty evaluation is required")
    for index, (row, sample) in enumerate(zip(rows, dataset), 1):
        require(row["index"] == index and row["images"] == sample["images"],
                f"Sample alignment mismatch at {index}")
        expected = json.loads(sample["messages"][1]["content"])
        require(valid_format(expected), f"Unsupported reference schema at {index}")
        require(row["expected"] == expected
                and row["prompt"] == sample["messages"][0]["content"],
                f"Reference or prompt mismatch at {index}")
        rescored = score_response(row["raw_response"], expected)
        require(row["scores"] == rescored, f"Saved v1 score mismatch at {index}")
    return run, [score_row(row) for row in rows]


def compare_runs(base, lora, metadata):
    require(len(base) == len(lora) > 0, "Prediction counts differ or are empty")
    slices = {}
    improved, regressed, shared_errors = [], [], []
    for left, right in zip(base, lora):
        require(all(left[key] == right[key] for key in ("index", "images", "prompt", "expected")),
                "Base and LoRA predictions are not paired")
        name = left["images"][0]
        require(name in metadata, f"Missing metadata for {name}")
        spec = metadata[name]
        require(spec["answer"] == left["expected"], f"Metadata answer mismatch for {name}")
        for dimension, label in slice_labels(spec).items():
            group = slices.setdefault(dimension, {}).setdefault(label, {"base": [], "lora": []})
            group["base"].append(left)
            group["lora"].append(right)
        a, b = left["normalized_exact_match"], right["normalized_exact_match"]
        if a != b or not a:
            destination = improved if b else regressed if a else shared_errors
            destination.append({
                "index": left["index"], "image": name, "expected": left["expected"],
                "base_normalized": left["normalized_object"],
                "lora_normalized": right["normalized_object"],
            })
    return {
        "overall": {"base": summarize(base), "lora": summarize(lora)},
        "slices": {dimension: {
            label: {model: summarize(rows) for model, rows in group.items()}
            for label, group in sorted(groups.items())
        } for dimension, groups in slices.items()},
        "paired_normalized_comparison": {
            "improved_count": len(improved), "regressed_count": len(regressed),
            "shared_error_count": len(shared_errors),
            "improved": improved, "regressed": regressed, "shared_errors": shared_errors,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--lora-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/nameplates_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists; choose a new filename: {args.output}")
    base_dir = args.base_dir or ROOT / f"results/nameplate_base_{args.split}_01"
    lora_dir = args.lora_dir or ROOT / f"results/nameplate_lora_{args.split}_01"
    dataset_path = args.dataset_dir / f"{args.split}.json"
    metadata_path = args.dataset_dir / "metadata.jsonl"
    base_run, base = load_run(base_dir, dataset_path)
    lora_run, lora = load_run(lora_dir, dataset_path)
    config = dict(lora_run["config"])
    require(bool(config.pop("adapter_name_or_path", None)), "LoRA run has no adapter")
    require(config == base_run["config"], "Inference configurations differ beyond the adapter")
    for key in ("seed", "dataset_sha256", "evaluator_sha256", "devices", "dtypes"):
        require(base_run[key] == lora_run[key], f"Evaluation protocols differ: {key}")
    require(base_run["split"] == lora_run["split"] == args.split, "Split mismatch")
    specs = [row for row in read_jsonl(metadata_path) if row["split"] == args.split]
    metadata = {row["image"]: row for row in specs}
    require(len(metadata) == len(specs) == len(base), "Metadata cardinality mismatch")
    sources = [dataset_path, metadata_path, Path(__file__),
               Path(__file__).with_name("evaluate_nameplates.py")]
    for folder in (base_dir, lora_dir):
        sources.extend([folder / "run.json", folder / "predictions.jsonl"])
    def source_name(path):
        resolved = path.resolve()
        return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)
    report = {
        "schema_version": 1, "split": args.split, "policy": POLICY,
        "source_sha256": {source_name(path): digest(path) for path in sources},
        **compare_runs(base, lora, metadata),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"split": args.split, "overall": report["overall"],
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
