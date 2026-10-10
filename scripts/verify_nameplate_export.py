"""Reload a merged model and compare selected outputs with saved adapter outputs."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def select_cases(rows):
    """Choose the first available example of each category, without new inference."""
    selected = []
    for correct, missing, name in [
        (True, False, "字段完整且回答正确"),
        (True, True, "含缺失字段且回答正确"),
        (False, True, "已知缺失字段错误"),
        (False, False, "已知非缺失字段错误"),
    ]:
        match = next((row for row in rows
                      if row["scores"]["strict_exact_match"] == correct
                      and any(value is None for value in row["expected"].values()) == missing), None)
        if match is not None:
            selected.append((name, match))
    if not selected:
        raise ValueError("参考预测中没有可用样例")
    return selected


def summarize(rows):
    return {
        "cases_checked": len(rows),
        "raw_output_matches": sum(row["raw_output_matches"] for row in rows),
        "finish_reason_matches": sum(row["finish_reason_matches"] for row in rows),
        "all_checks_passed": bool(rows) and all(
            row["raw_output_matches"] and row["finish_reason_matches"] for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/nameplate_merged_01")
    parser.add_argument("--reference-dir", default="outputs/nameplate_lora_test_01")
    parser.add_argument("--output", default="logs/nameplate_reload_check_01.json")
    args = parser.parse_args()
    model_path = (ROOT / args.model).resolve()
    reference_dir = (ROOT / args.reference_dir).resolve()
    output = ROOT / args.output
    if output.exists():
        parser.error(f"报告已存在，请用 --output 指定新文件：{output}")
    if not (model_path / "model.safetensors.index.json").is_file():
        parser.error(f"导出目录中未找到权重索引：{model_path}")

    reference_run = json.loads((reference_dir / "run.json").read_text())
    if not reference_run["config"].get("adapter_name_or_path"):
        parser.error("参考记录必须来自基础模型加 LoRA 的推理结果")
    predictions_bytes = (reference_dir / "predictions.jsonl").read_bytes()
    references = [json.loads(line) for line in predictions_bytes.splitlines() if line.strip()]
    if len(references) != reference_run["sample_count"]:
        parser.error("参考预测条数与运行记录不一致，请先确认原评估完成")
    selected = select_cases(references)

    # Reuse the inference protocol, but load the complete exported model alone.
    config = dict(reference_run["config"])
    config.pop("adapter_name_or_path")
    config["model_name_or_path"] = str(model_path)
    if config.get("do_sample") is not False:
        parser.error("本检查要求参考评估关闭随机采样")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch
    from llamafactory.chat import ChatModel

    torch.manual_seed(reference_run["seed"])
    print(f"独立加载导出模型：{model_path}", flush=True)
    print(f"对比 {len(selected)} 条固定样例；不加载 adapter。", flush=True)
    model = ChatModel(config)
    generation = model.engine.model.generation_config
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None

    rows = []
    for category, reference in selected:
        started = time.monotonic()
        # The reference output and gold answer are comparison data, never inputs.
        response = model.chat(
            messages=[{"role": "user", "content": reference["prompt"]}],
            images=[str(ROOT / "data" / image) for image in reference["images"]],
            do_sample=False,
            max_new_tokens=config["max_new_tokens"],
        )[0]
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        row = {
            "category": category,
            "index": reference["index"],
            "images": reference["images"],
            "prompt": reference["prompt"],
            "expected": reference["expected"],
            "reference_raw_response": reference["raw_response"],
            "reloaded_raw_response": response.response_text,
            "reference_finish_reason": reference["finish_reason"],
            "reloaded_finish_reason": response.finish_reason,
            "raw_output_matches": response.response_text == reference["raw_response"],
            "finish_reason_matches": response.finish_reason == reference["finish_reason"],
            "seconds": round(time.monotonic() - started, 3),
        }
        rows.append(row)
        print(
            f"第 {row['index']} 条（{category}）："
            f"原始回答一致={row['raw_output_matches']}，"
            f"终止原因一致={row['finish_reason_matches']}",
            flush=True,
        )

    summary = summarize(rows)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "reference_dir": str(reference_dir),
        "reference_predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config,
        "seed": reference_run["seed"],
        "devices": sorted({str(p.device) for p in model.engine.model.parameters()}),
        "dtypes": sorted({str(p.dtype) for p in model.engine.model.parameters()}),
        "summary": summary,
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"报告已保存：{output}", flush=True)
    if not summary["all_checks_passed"]:
        raise SystemExit("存在输出差异，请保留报告，核对推理设置与合并精度。")


if __name__ == "__main__":
    main()
