"""Validate a completed v2 run and archive results/adapters locally (no upload).

Refuses incomplete runs or existing destinations. The original runtime files are
preserved, and every copied file records both source and publication SHA-256.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from evaluate_nameplates import score_response, summarize as strict_summary
from analyze_nameplate_predictions import score_row, summarize
from run_nameplate_v2_experiment import selection_key, grouped_test_report

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_prediction_records(rows, samples):
    require(len(rows) == len(samples) > 0, "Prediction count mismatch")
    for index, (row, sample) in enumerate(zip(rows, samples), 1):
        require(row["index"] == index and row["images"] == sample["images"], "Prediction alignment mismatch")
        expected = json.loads(sample["messages"][1]["content"])
        require(row["expected"] == expected and row["prompt"] == sample["messages"][0]["content"],
                "Reference or prompt differs")
        require(row["scores"] == score_response(row["raw_response"], expected), "Saved score differs from recomputation")
    return summarize([score_row(row) for row in rows])


def verify_evaluation(folder, dataset_path):
    run = read(folder / "run.json")
    require(run["status"] == "completed", f"Incomplete inference: {folder}")
    require(run["dataset_sha256"] == digest(dataset_path), "Dataset hash differs")
    require(run["scorer_sha256"] == digest(ROOT / "scripts/evaluate_nameplates.py"), "Scorer version differs")
    require(run["evaluator_sha256"] == digest(ROOT / "scripts/evaluate_nameplates_v2.py"), "Evaluator version differs")
    samples = read(dataset_path)
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines() if line]
    require(len(rows) == run["sample_count"], "Run sample count differs")
    analysis = verify_prediction_records(rows, samples)
    summary = strict_summary(rows)
    summary["generated_tokens_limit_count"] = sum(row["finish_reason"] == "length" for row in rows)
    require(summary == read(folder / "summary.json"), "Strict summary differs")
    require(analysis == read(folder / "analysis.json"), "Diagnostic summary differs")
    return analysis


def validate_run(run_dir):
    require(read(run_dir / "status.json")["status"] == "completed", "Experiment is not complete")
    protocol = read(run_dir / "protocol.json")
    for name, checksum in protocol["sha256"].items():
        require(digest(ROOT / name) == checksum, f"Frozen file changed: {name}")
    selection = read(run_dir / "selection.json")
    require(selection["protocol_sha256"] == digest(run_dir / "protocol.json"), "Selection protocol differs")
    final = read(run_dir / "final_summary.json")
    require(final["protocol"] == protocol and final["selection"] == selection["selected"], "Final protocol/selection differs")
    candidates = read(run_dir / "validation_candidates.json")
    for arm in ("control", "hard"):
        audit = read(run_dir / f"train_{arm}/training_audit.json")
        require(audit["completed_steps"] == 150 and audit["changed_tensors"] == 112
                and audit["all_adapter_values_finite"], "Training audit failed")
        require([candidate["step"] for candidate in candidates[arm]] == protocol["candidate_steps"],
                "Candidate checkpoint set differs")
        for candidate in candidates[arm]:
            folder = run_dir / f"{arm}_step{candidate['step']}_val"
            require(verify_evaluation(folder, ROOT / "data/nameplates_v2/val.json") == candidate["metrics"],
                    "Candidate metric mismatch")
        chosen = min(candidates[arm], key=selection_key)
        require(chosen == selection["selected"][arm], "Checkpoint choice differs from preregistered rule")
        require(digest(ROOT / chosen["adapter"] / "adapter_model.safetensors") == chosen["adapter_sha256"],
                "Selected adapter changed")
        regression = verify_evaluation(run_dir / f"{arm}_v1_val", ROOT / "data/nameplates_v1/val.json")
        require(regression == final["v1_validation_regression"][arm], "Regression summary mismatch")
    frozen_at = datetime.fromisoformat(selection["frozen_at"])
    base_fingerprint = read(run_dir / "base_test/run.json")["model_sha256"]
    for name in ("base", "v1_adapter", "control", "hard"):
        folder = run_dir / f"{name}_test"
        run = read(folder / "run.json")
        require(run["model_sha256"] == base_fingerprint, "Test base model differs across arms")
        require(datetime.fromisoformat(run["started_at"]) >= frozen_at, "Test began before checkpoint selection")
        metrics = verify_evaluation(folder, ROOT / "data/nameplates_v2/test.json")
        require(metrics == final["test"][name], "Final test summary mismatch")
        if name in selection["selected"]:
            require(run["adapter_sha256"]["adapter_model.safetensors"] == selection["selected"][name]["adapter_sha256"],
                    "Test used a different checkpoint")
        elif name == "base":
            require(run["adapter_sha256"] is None, "Base-model test unexpectedly loaded an adapter")
        else:
            require(run["adapter_sha256"]["adapter_model.safetensors"] ==
                    digest(ROOT / "artifacts/nameplate_lora_01/adapter_model.safetensors"),
                    "v1 reference adapter differs")
    require(grouped_test_report(run_dir) == final["grouped_test"], "Grouped analysis differs")
    verify_evaluation(run_dir / "merged_v1_test", ROOT / "data/nameplates_v1/test.json")
    require(read(run_dir / "merged_v1_test/comparison.json")["all_checks_passed"], "Merged comparison failed")
    return final


def report_text(final):
    lines = ["# v2 数据扩充对照结果", "", "两组均从相同基础模型训练，seed=42，800 条数据、150 次更新。"
             "checkpoint 仅根据验证集选择，随后固定权重进行最终测试。", "",
             "## 最终测试", "", "测试集包含 200 张合成图片、50 个父组。", "",
             "|模型|严格全对|规则处理后全对|缺失字段正确|有值字段误置空|",
             "|---|---:|---:|---:|---:|"]
    labels = {"base": "基础模型", "v1_adapter": "原 v1 adapter", "control": "普通扩充", "hard": "困难扩充"}
    for name, label in labels.items():
        m = final["test"][name]
        null = m["null_metrics_fence_only"]
        lines.append(f"|{label}|{m['counts']['strict_exact_match']}/{m['sample_count']}|"
                     f"{m['counts']['normalized_exact_match']}/{m['sample_count']}|"
                     f"{null['tp']}/{null['target_null_count']}|{null['fp']}/{null['target_present_count']}|")
    delta = final["grouped_test"]["hard_minus_control"]
    low, high = delta["group_bootstrap_95_percentile_interval"]
    lines += ["", f"困难组减普通组的严格全对率差为 {delta['strict_exact_rate_difference'] * 100:+.1f} 个百分点；"
              f"按父组配对 bootstrap 的 95% 区间为 [{low * 100:+.1f}, {high * 100:+.1f}] 个百分点。",
              "该区间只反映这批合成父组的抽样变化，不包括训练随机种子的变化。", "",
              "## 所选 checkpoint 与旧验证集回归", "",
              "|训练组|所选步数|v2 验证集严格全对|v1 验证集严格全对|", "|---|---:|---:|---:|"]
    for arm in ("control", "hard"):
        selected = final["selection"][arm]
        old = final["v1_validation_regression"][arm]
        lines.append(f"|{labels[arm]}|{selected['step']}|{selected['metrics']['counts']['strict_exact_match']}/200|"
                     f"{old['counts']['strict_exact_match']}/100|")
    lines += ["", "## 证据与范围", "",
              "- `final_summary.json`：总表、切片统计、整组四图全对数及配对区间。",
              "- `protocol.json` / `selection.json`：预先固定的选择规则、时间和权重指纹。",
              "- 各评估子目录：原始预测、运行配置和评分；`training`：训练记录与曲线。",
              "- `publication_manifest.json`：每份源文件与发布副本的哈希。",
              "- 普通/困难扩充比较的是完整数据配方，不能分别估计缺失比例和成对构造的独立贡献。",
              "- 只有一个训练种子，全部为合成图片。尚未开展真实照片测试。", ""]
    return "\n".join(lines)


def publish(run_dir, destination, artifacts):
    require(not destination.exists() and not artifacts.exists(), "Publication destination already exists")
    final = validate_run(run_dir)
    destination.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    copied = []

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        published = data.replace((str(ROOT) + "/").encode(), b"") if source.suffix in (".json", ".jsonl") else data
        with target.open("xb") as handle:
            handle.write(published)
        copied.append({"source": str(source.relative_to(ROOT)), "published": str(target.relative_to(ROOT)),
                       "source_sha256": hashlib.sha256(data).hexdigest(), "published_sha256": digest(target)})

    for path in sorted(run_dir.glob("*.json")):
        if path.name != "status.json":
            copy(path, destination / path.name)
    for folder in sorted(run_dir.iterdir()):
        if folder.is_dir() and (folder / "predictions.jsonl").is_file():
            for path in sorted(folder.iterdir()):
                if path.suffix in (".json", ".jsonl"):
                    copy(path, destination / folder.name / path.name)
    for arm in ("control", "hard"):
        for name in ("train_results.json", "training_audit.json", "trainer_state.json", "trainer_log.jsonl", "training_loss.png"):
            copy(run_dir / f"train_{arm}" / name, destination / "training" / arm / name)
        checkpoint = ROOT / final["selection"][arm]["adapter"]
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            copy(checkpoint / name, artifacts / arm / name)
        copy(ROOT / "artifacts/nameplate_lora_01/LICENSE", artifacts / arm / "LICENSE")
        metrics = final["test"][arm]
        (artifacts / arm / "README.md").write_text(
            f"# v2 {arm} LoRA adapter\n\n"
            f"Selected on v2 validation at step {final['selection'][arm]['step']}. "
            f"Synthetic test strict exact: {metrics['counts']['strict_exact_match']}/200.\n\n"
            "Requires Qwen/Qwen2-VL-2B-Instruct revision 895c3a49bc3fa70a340399125c650a463535e71c "
            "and its original tokenizer/processor. rank=4, alpha=8, q_proj/v_proj, BF16 inference.\n\n"
            "Single training seed; synthetic data only. See the experiment protocol and raw predictions "
            f"in results/{destination.name}. Base-model license is included.\n", encoding="utf-8")
    (destination / "REPORT.md").write_text(report_text(final), encoding="utf-8")
    manifest = {"description": "Local publication only; original runtime artifacts preserved. "
                 "Only repository absolute path prefixes in JSON/JSONL were removed.",
                "publisher_sha256": digest(Path(__file__)), "files": copied,
                "generated_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                                     [destination / "REPORT.md", *(artifacts / arm / "README.md" for arm in ("control", "hard"))]}}
    (destination / "publication_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for item in copied:
        require(digest(ROOT / item["published"]) == item["published_sha256"], "Publication hash mismatch")
    return {"copied_files": len(copied), "report": str(destination / "REPORT.md"), "artifacts": str(artifacts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/v2_experiment_01")
    parser.add_argument("--destination", type=Path, default=ROOT / "results/v2_experiment_01")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/nameplate_v2_01")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for the current run; stop on failure or a dead runner")
    args = parser.parse_args()
    if args.wait:
        previous = None
        while True:
            state = read(args.run_dir / "status.json")
            if state["status"] == "completed":
                break
            require(state["status"] == "running", f"Experiment stopped: {state.get('error', state['status'])}")
            try:
                os.kill(state["pid"], 0)
            except ProcessLookupError as error:
                raise RuntimeError("Experiment runner exited before completion") from error
            if state["current_stage"] != previous:
                previous = state["current_stage"]
                print(f"Waiting for experiment: {previous}", flush=True)
            time.sleep(30)
    if args.check_only:
        validate_run(args.run_dir.resolve())
        print("Complete run verified without copying files.")
    else:
        print(json.dumps(publish(args.run_dir.resolve(), args.destination.resolve(), args.artifacts.resolve()),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
