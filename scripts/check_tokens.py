"""Inspect the native SFT preprocessing without loading model weights."""
import argparse
import json
import os
from pathlib import Path


def inspect_split(dataset, tokenizer, cutoff_len):
    rows = []
    for row in dataset:
        rows.append({
            "tokens": len(row["input_ids"]),
            "answer_tokens": sum(token != -100 for token in row["labels"]),
            "images": len(row.get("images") or []),
            "label_length_matches_input": len(row["labels"]) == len(row["input_ids"]),
        })
    supervised = [token for token in dataset[0]["labels"] if token != -100] if rows else []
    summary = {
        "samples": len(rows),
        "min_tokens": min((row["tokens"] for row in rows), default=0),
        "max_tokens": max((row["tokens"] for row in rows), default=0),
        # Reaching the limit is a warning, not proof that truncation occurred.
        "at_or_above_cutoff": sum(row["tokens"] >= cutoff_len for row in rows),
        "empty_answer_labels": sum(row["answer_tokens"] == 0 for row in rows),
        "label_length_mismatches": sum(not row["label_length_matches_input"] for row in rows),
        "first_supervised_answer": tokenizer.decode(
            supervised, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ),
    }
    return rows, summary


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_mps.yaml")
    parser.add_argument("--output", default="logs/token_validation.json")
    args = parser.parse_args()
    output = root / args.output
    if output.exists():
        parser.error(f"报告已存在，请用 --output 指定新文件：{output}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import yaml
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_template_and_fix_tokenizer, get_dataset

    config = yaml.safe_load((root / args.config).read_text())
    model_args, data_args, training_args, _, _ = get_train_args(config)
    if data_args.packing:
        parser.error("本检查按独立样本统计，请使用 packing: false")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    datasets = get_dataset(
        template, model_args, data_args, training_args, stage="sft", **tokenizer_module
    )
    summary = {}
    details = {}
    for name, dataset in datasets.items():
        if dataset is not None:
            details[name], summary[name] = inspect_split(dataset, tokenizer, data_args.cutoff_len)

    report = {"config": args.config, "cutoff_len": data_args.cutoff_len,
              "summary": summary, "details": details}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"报告已保存：{output}", flush=True)
    if not summary or any(
        not item["samples"] or item["at_or_above_cutoff"]
        or item["empty_answer_labels"] or item["label_length_mismatches"]
        for item in summary.values()
    ):
        raise SystemExit("发现空数据、长度边界或标签问题，请先检查报告。")


if __name__ == "__main__":
    main()
