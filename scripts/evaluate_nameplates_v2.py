"""Evaluate a configurable model/dataset; optionally compare every exported output.

The frozen v1 scorer is reused. Reference predictions never enter model inputs.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluate_nameplates import ROOT, score_response, summarize, valid_format
from analyze_nameplate_predictions import score_row, summarize as summarize_analysis


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def artifact_hashes(path):
    paths = sorted(path.glob("*.json")) + sorted(path.glob("*.safetensors"))
    if not paths:
        raise ValueError(f"No model artifacts found at {path}")
    return {p.name: sha256(p) for p in paths}


def compare_predictions(rows, references):
    if len(rows) != len(references) or not rows:
        raise ValueError("Reference and generated prediction counts differ or are empty")
    comparisons = []
    for row, reference in zip(rows, references):
        for key in ("index", "images", "prompt", "expected"):
            if row[key] != reference[key]:
                raise ValueError(f"Reference alignment mismatch: {key}, index {row['index']}")
        comparisons.append({
            "index": row["index"], "images": row["images"],
            "raw_output_matches": row["raw_response"] == reference["raw_response"],
            "finish_reason_matches": row["finish_reason"] == reference["finish_reason"],
            "scores_match": row["scores"] == reference["scores"],
        })
    return {
        "cases_checked": len(comparisons),
        "raw_output_matches": sum(r["raw_output_matches"] for r in comparisons),
        "finish_reason_matches": sum(r["finish_reason_matches"] for r in comparisons),
        "scores_match": sum(r["scores_match"] for r in comparisons),
        "all_checks_passed": all(r["raw_output_matches"] and r["finish_reason_matches"]
                                 and r["scores_match"] for r in comparisons),
        "differences": [r for r in comparisons if not all(
            r[key] for key in ("raw_output_matches", "finish_reason_matches", "scores_match"))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/nameplates_v1")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--require-match", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    if args.limit < 0 or (args.require_match and args.reference_dir is None):
        parser.error("Invalid limit or missing reference directory")
    dataset_path = args.dataset_dir.resolve() / f"{args.split}.json"
    samples = json.loads(dataset_path.read_text())
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        parser.error("Empty evaluation dataset")
    image_hashes = {}
    for sample in samples:
        if not valid_format(json.loads(sample["messages"][1]["content"])):
            parser.error("Labels must use the current integer V/W schema")
        for name in sample["images"]:
            path = (ROOT / "data" / name).resolve()
            if not path.is_relative_to((ROOT / "data").resolve()):
                parser.error("Image path is outside the data directory")
            image_hashes[name] = sha256(path)
    config = {
        "model_name_or_path": str(args.model.resolve()), "template": "qwen2_vl",
        "infer_dtype": "bfloat16", "infer_backend": "huggingface", "flash_attn": "disabled",
        "trust_remote_code": False, "image_max_pixels": 301056, "image_min_pixels": 3136,
        "do_sample": False, "max_new_tokens": 128,
    }
    if args.adapter:
        config["adapter_name_or_path"] = str(args.adapter.resolve())
    references = None
    if args.reference_dir:
        reference_run = json.loads((args.reference_dir / "run.json").read_text())
        references = [json.loads(line) for line in
                      (args.reference_dir / "predictions.jsonl").read_text().splitlines() if line]
        if reference_run["dataset_sha256"] != sha256(dataset_path):
            parser.error("Reference dataset fingerprint mismatch")
        if len(references) != len(samples) or reference_run["sample_count"] != len(samples):
            parser.error("Reference must cover all evaluated samples")
        for key, value in config.items():
            if key not in ("model_name_or_path", "adapter_name_or_path"):
                if reference_run["config"].get(key) != value:
                    parser.error(f"Reference inference protocol mismatch: {key}")
        if reference_run["seed"] != 42:
            parser.error("Reference seed mismatch")
    run = {
        "started_at": datetime.now(timezone.utc).isoformat(), "split": args.split,
        "sample_count": len(samples), "dataset_sha256": sha256(dataset_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "scorer_sha256": sha256(Path(__file__).with_name("evaluate_nameplates.py")),
        "analysis_sha256": sha256(Path(__file__).with_name("analyze_nameplate_predictions.py")),
        "image_sha256": image_hashes, "seed": 42, "config": config,
        "model_sha256": artifact_hashes(args.model.resolve()),
        "adapter_sha256": artifact_hashes(args.adapter.resolve()) if args.adapter else None,
        "status": "running",
    }
    if args.reference_dir:
        run["reference_predictions_sha256"] = sha256(args.reference_dir / "predictions.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump(args.output_dir / "run.json", run)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch
    from llamafactory.chat import ChatModel
    torch.manual_seed(42)
    model = ChatModel(config)
    generation = model.engine.model.generation_config
    generation.do_sample = False
    generation.temperature = generation.top_p = generation.top_k = None
    run["devices"] = sorted({str(p.device) for p in model.engine.model.parameters()})
    run["dtypes"] = sorted({str(p.dtype) for p in model.engine.model.parameters()})
    dump(args.output_dir / "run.json", run)
    rows = []
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            start = time.monotonic()
            response = model.chat(messages=[sample["messages"][0]],
                                  images=[str(ROOT / "data" / name) for name in sample["images"]],
                                  do_sample=False, max_new_tokens=128)[0]
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            expected = json.loads(sample["messages"][1]["content"])
            row = {
                "index": index, "images": sample["images"],
                "prompt": sample["messages"][0]["content"], "expected": expected,
                "raw_response": response.response_text, "finish_reason": response.finish_reason,
                "response_length": response.response_length,
                "seconds": round(time.monotonic() - start, 3),
                "scores": score_response(response.response_text, expected),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            print(f"[{index}/{len(samples)}] strict={row['scores']['strict_exact_match']} "
                  f"seconds={row['seconds']}", flush=True)
    summary = summarize(rows)
    summary["generated_tokens_limit_count"] = sum(r["finish_reason"] == "length" for r in rows)
    dump(args.output_dir / "summary.json", summary)
    dump(args.output_dir / "analysis.json", summarize_analysis([score_row(r) for r in rows]))
    matches = True
    if references is not None:
        comparison = compare_predictions(rows, references)
        dump(args.output_dir / "comparison.json", comparison)
        matches = comparison["all_checks_passed"]
        print(json.dumps(comparison, ensure_ascii=False), flush=True)
    run.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
    dump(args.output_dir / "run.json", run)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.require_match and not matches:
        raise SystemExit("Exported model differs; inspect comparison.json")


if __name__ == "__main__":
    main()
