"""Check published data, hashes and saved scores without loading a model."""

import hashlib
import itertools
import json
from pathlib import Path

from PIL import Image

from evaluate_nameplates import score_response, summarize, valid_format
from verify_nameplate_export import select_cases, summarize as summarize_reload


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/nameplates_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_data():
    manifest = read_json(DATA / "manifest.json")
    for filename, digest in manifest["files_sha256"].items():
        require(sha256(DATA / filename) == digest, f"数据文件哈希不符：{filename}")
    require(sha256(ROOT / "scripts/generate_nameplates.py") == manifest["generator_sha256"],
            "生成脚本与实验版本不同")
    metadata = read_jsonl(DATA / "metadata.jsonl")
    by_image = {row["image"]: row for row in metadata}
    require(len(by_image) == len(metadata) == 600, "生成记录数量或图片唯一性错误")
    datasets, groups, referenced = {}, {}, set()
    registry = read_json(ROOT / "data/dataset_info.json")
    for split, count in [("train", 400), ("val", 100), ("test", 100)]:
        rows = read_json(DATA / f"{split}.json")
        datasets[split] = rows
        require(len(rows) == count, f"{split} 样本数量错误")
        require(registry[f"nameplate_{split}"]["file_name"] == f"nameplates_v1/{split}.json",
                f"{split} 数据注册错误")
        group = {"image_hash": set(), "model_group": set(), "template": set()}
        for i, row in enumerate(rows, 1):
            messages = row["messages"]
            require(len(messages) == 2 and [m["role"] for m in messages] == ["user", "assistant"],
                    f"{split}/{i} 消息结构错误")
            require(messages[0]["content"] == manifest["prompt"], f"{split}/{i} 提示不同")
            require(len(row["images"]) == messages[0]["content"].count("<image>") == 1,
                    f"{split}/{i} 图片占位符不匹配")
            answer = json.loads(messages[1]["content"])
            require(valid_format(answer), f"{split}/{i} 答案格式错误")
            name = row["images"][0]
            require(name not in referenced, f"重复图片引用：{name}")
            referenced.add(name)
            spec = by_image[name]
            require(spec["split"] == split and spec["answer"] == answer,
                    f"{split}/{i} 标签或生成集合不一致")
            path = (ROOT / "data" / name).resolve()
            require(path.is_relative_to(DATA.resolve()), f"数据目录外的图片：{name}")
            digest = sha256(path)
            require(digest == spec["image_sha256"], f"图片哈希不符：{name}")
            with Image.open(path) as image:
                image.load()
                require(image.size == tuple(manifest["image_size"]), f"图片尺寸错误：{name}")
            group["image_hash"].add(digest)
            for field in ("model_group", "template"):
                group[field].add(spec[field])
        groups[split] = group
    require(referenced == set(by_image), "存在未引用的生成记录")
    require({str(p.relative_to(ROOT / "data")) for p in (DATA / "images").glob("*.png")}
            == referenced, "图片目录与数据引用不一致")
    for left, right in itertools.combinations(groups, 2):
        for field in groups[left]:
            require(not groups[left][field] & groups[right][field],
                    f"{left}/{right} 存在重复 {field}")
    print("数据：400/100/100；600 张图片解码与哈希通过；跨集合图片、型号、模板重复为 0")
    return datasets


def check_evaluations(datasets):
    runs, predictions = {}, {}
    for split in ("val", "test"):
        for model in ("base", "lora"):
            name = f"nameplate_{model}_{split}_01"
            folder = ROOT / "results" / name
            run = read_json(folder / "run.json")
            rows = read_jsonl(folder / "predictions.jsonl")
            require(run["split"] == split and run["sample_count"] == len(rows) == 100,
                    f"{name} 样本数量或集合不符")
            require(run["dataset_sha256"] == sha256(DATA / f"{split}.json"), f"{name} 数据版本不同")
            require(run["evaluator_sha256"] == sha256(ROOT / "scripts/evaluate_nameplates.py"),
                    f"{name} 评分脚本版本不同")
            for i, (row, sample) in enumerate(zip(rows, datasets[split]), 1):
                require(row["index"] == i and row["images"] == sample["images"], f"{name}/{i} 图片错位")
                require(row["prompt"] == sample["messages"][0]["content"] and
                        row["expected"] == json.loads(sample["messages"][1]["content"]),
                        f"{name}/{i} 提示或标准答案不一致")
                require(row["scores"] == score_response(row["raw_response"], row["expected"]),
                        f"{name}/{i} 保存评分与重算不一致")
            summary = summarize(rows)
            summary["generated_tokens_limit_count"] = sum(r["finish_reason"] == "length" for r in rows)
            require(summary == read_json(folder / "summary.json"), f"{name} 汇总与重算不一致")
            runs[model] = run
            predictions[name] = rows
        base, lora = runs["base"], runs["lora"]
        config = dict(lora["config"])
        require(bool(config.pop("adapter_name_or_path", None)), "LoRA 运行未记录 adapter")
        require(config == base["config"] and base["seed"] == lora["seed"] == 42,
                f"{split} 基线与 LoRA 协议不同")
    print("评分：4 组、400 条原始预测重算通过；同集合基线与 LoRA 设置可比")
    return predictions


def check_publication(predictions):
    manifest = read_json(ROOT / "results/publication_manifest.json")
    for item in manifest["files"]:
        require(sha256(ROOT / item["published"]) == item["published_sha256"],
                f"发布文件哈希不符：{item['published']}")
    expected = {
        "adapter_model.safetensors": "2d5397a9cacf0b5ad50ab05c94125832a0712a9db1ded7b9b51fe3bece52ba53",
        "adapter_config.json": "03a782b15bdda808340b1888574a41655910c8108aa08b9f187fb2c911a356c1",
    }
    for name, digest in expected.items():
        require(sha256(ROOT / "artifacts/nameplate_lora_01" / name) == digest, f"adapter 版本不符：{name}")
    report = read_json(ROOT / "results/checks/nameplate_reload_check_01.json")
    require(report["reference_predictions_sha256"] ==
            sha256(ROOT / "results/nameplate_lora_test_01/predictions.jsonl"), "重载参考预测哈希不符")
    require(report["verifier_sha256"] == sha256(ROOT / "scripts/verify_nameplate_export.py"),
            "重载检查脚本版本不同")
    require("adapter_name_or_path" not in report["config"], "独立重载不应另外加载 adapter")
    cases = report["cases"]
    selected = select_cases(predictions["nameplate_lora_test_01"])
    require(len(cases) == len(selected) == 4, "重载样例数量不同")
    for case, (category, reference) in zip(cases, selected):
        require(case["category"] == category and case["index"] == reference["index"], "重载样例选择不同")
        require(case["reference_raw_response"] == reference["raw_response"] and
                case["reference_finish_reason"] == reference["finish_reason"], "重载引用不符")
        require(case["raw_output_matches"] == (case["reference_raw_response"] == case["reloaded_raw_response"]),
                "重载原始回答比较不符")
        require(case["finish_reason_matches"] ==
                (case["reference_finish_reason"] == case["reloaded_finish_reason"]), "重载终止原因比较不符")
    require(summarize_reload(cases) == report["summary"] and report["summary"]["all_checks_passed"],
            "重载汇总不符或存在失败")
    print(f"发布：{len(manifest['files'])} 份副本哈希通过；adapter 指纹一致；4 条重载记录一致")


def main():
    datasets = check_data()
    predictions = check_evaluations(datasets)
    check_publication(predictions)
    print("全部检查通过。本检查未加载模型或重新生成预测。")


if __name__ == "__main__":
    main()
