"""Evaluate original nameplate responses; fence removal is diagnostic only.

Run this script yourself from the lab directory. Without --adapter it loads the
base model. Keep the same settings when comparing a future LoRA adapter.
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("型号", "电压", "功率")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复 JSON 键：{key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"不是标准 JSON 常量：{value}")


def parse_object(text):
    """Parse the whole response, accepting JSON whitespace but no extra prose."""
    try:
        value = json.loads(
            text, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def valid_format(value):
    if value is None or set(value) != set(FIELDS):
        return False
    for field in FIELDS:
        item = value[field]
        if item is None:
            continue
        if not isinstance(item, str) or not item.strip():
            return False
        if item.strip().upper() in {"NULL", "--", "N/A"}:
            return False
        if field in ("电压", "功率"):
            unit = "V" if field == "电压" else "W"
            # This dataset uses integer readings, with no unit conversions.
            if re.fullmatch(r"[0-9]+" + unit, item) is None:
                return False
    return True


def score_response(raw, expected):
    original = parse_object(raw)
    diagnostic = original
    fence_removed = False
    if original is None:
        # Only remove a single surrounding code fence. Do not extract a JSON
        # substring from explanatory prose or fix values, keys, or units.
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), re.S | re.I)
        if match:
            diagnostic = parse_object(match.group(1))
            fence_removed = True
    compliant = valid_format(original)
    field_correct = {
        field: diagnostic is not None
        and field in diagnostic
        and diagnostic[field] == expected[field]
        for field in FIELDS
    }
    return {
        "raw_json_object": original is not None,
        "strict_format": compliant,
        "strict_exact_match": compliant and original == expected,
        "fence_removed_for_diagnosis": fence_removed,
        "diagnostic_exact_match": diagnostic == expected,
        "diagnostic_field_correct": field_correct,
    }


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError("评估数据为空")
    metrics = {
        "sample_count": n,
        "fence_removed_count": sum(
            r["scores"]["fence_removed_for_diagnosis"] for r in rows
        ),
    }
    for name in (
        "raw_json_object", "strict_format", "strict_exact_match", "diagnostic_exact_match"
    ):
        metrics[name + "_rate"] = sum(r["scores"][name] for r in rows) / n
    # Every sample is in the denominator; unparseable answers count as wrong.
    metrics["diagnostic_field_accuracy"] = {
        field: sum(r["scores"]["diagnostic_field_correct"][field] for r in rows) / n
        for field in FIELDS
    }
    missing = [
        r["scores"]["diagnostic_field_correct"][field]
        for r in rows for field in FIELDS if r["expected"][field] is None
    ]
    metrics["missing_field_count"] = len(missing)
    metrics["diagnostic_missing_value_accuracy"] = (
        sum(missing) / len(missing) if missing else None
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=0, help="0 表示整个集合")
    parser.add_argument("--adapter", help="留空即评估基础模型")
    parser.add_argument("--output-dir", required=True, help="使用新的目录，保留旧结果")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit 不能小于 0")

    dataset_path = ROOT / "data/nameplates_v1" / f"{args.split}.json"
    dataset_bytes = dataset_path.read_bytes()
    samples = json.loads(dataset_bytes)
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        parser.error("评估集合为空")
    output_dir = ROOT / args.output_dir
    if output_dir.exists():
        parser.error(f"输出目录已存在，请使用新名字：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch
    from llamafactory.chat import ChatModel

    torch.manual_seed(42)
    config = {
        "model_name_or_path": str(ROOT / "models/Qwen2-VL-2B-Instruct"),
        "template": "qwen2_vl",
        "infer_dtype": "bfloat16",
        "infer_backend": "huggingface",
        "flash_attn": "disabled",
        "trust_remote_code": False,
        "image_max_pixels": 672 * 448,
        "image_min_pixels": 3136,
        "do_sample": False,
        "max_new_tokens": 128,
    }
    if args.adapter:
        config["adapter_name_or_path"] = str(ROOT / args.adapter)
    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "sample_count": len(samples),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": 42,
        "config": config,
        "diagnostic_rule": "仅移除包住整段回答的代码围栏；不修改字段或数值",
    }
    (output_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"加载模型；本次评估 {args.split} 的 {len(samples)} 条数据。", flush=True)
    model = ChatModel(config)
    generation = model.engine.model.generation_config
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    run["devices"] = sorted({str(p.device) for p in model.engine.model.parameters()})
    run["dtypes"] = sorted({str(p.dtype) for p in model.engine.model.parameters()})
    (output_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    with (output_dir / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            expected = json.loads(sample["messages"][1]["content"])
            images = [str(ROOT / "data" / path) for path in sample["images"]]
            start = time.monotonic()
            # The reference assistant answer is never passed to the model.
            response = model.chat(
                messages=[sample["messages"][0]], images=images,
                do_sample=False, max_new_tokens=128,
            )[0]
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            raw = response.response_text
            scores = score_response(raw, expected)
            row = {
                "index": index,
                "images": sample["images"],
                "prompt": sample["messages"][0]["content"],
                "raw_response": raw,
                "expected": expected,
                "finish_reason": response.finish_reason,
                "response_length": response.response_length,
                "seconds": round(time.monotonic() - start, 3),
                "scores": scores,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            print(
                f"[{index}/{len(samples)}] 格式合规={scores['strict_format']} "
                f"严格全对={scores['strict_exact_match']} "
                f"诊断全对={scores['diagnostic_exact_match']} 用时={row['seconds']}s",
                flush=True,
            )

    summary = summarize(rows)
    summary["generated_tokens_limit_count"] = sum(
        r["finish_reason"] == "length" for r in rows
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"结果已保存：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
