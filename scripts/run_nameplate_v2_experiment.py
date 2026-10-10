"""Run the preregistered v2 experiment serially and persist progress and results.

Each child owns the MPS device until it exits. Failures stop the queue and remain
in status.json. This runner never silently resumes an incomplete training run.
"""
import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from analyze_nameplate_predictions import score_row, summarize

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def predictions(path):
    return [score_row(json.loads(line)) for line in (path / "predictions.jsonl").read_text().splitlines() if line]


def selection_key(candidate):
    metrics = candidate["metrics"]
    nulls = metrics["null_metrics_fence_only"]
    return (-metrics["counts"]["strict_exact_match"], nulls["spurious_value_count"],
            nulls["fp"], candidate["step"])


def grouped_test_report(run_dir):
    metadata = {r["image"]: r for r in (
        json.loads(line) for line in (ROOT / "data/nameplates_v2/metadata.jsonl").read_text().splitlines())
        if r["split"] == "test"}
    models = {name: predictions(run_dir / f"{name}_test")
              for name in ("base", "v1_adapter", "control", "hard")}
    report = {"slices": {}, "group_exact": {}}
    group_scores = {}
    for name, rows in models.items():
        slices, groups = {}, {}
        for row in rows:
            spec = metadata[row["images"][0]]
            for dimension in ("font_novelty", "value_novelty", "missing_mode", "focus_field"):
                label = str(spec[dimension] or "none")
                slices.setdefault(dimension, {}).setdefault(label, []).append(row)
            groups.setdefault(spec["parent_group"], []).append(row)
        report["slices"][name] = {dimension: {label: summarize(items) for label, items in labels.items()}
                                   for dimension, labels in slices.items()}
        report["group_exact"][name] = {
            "groups": len(groups),
            "all_four_correct": sum(all(r["scores"]["strict_exact_match"] for r in items)
                                    for items in groups.values()),
        }
        group_scores[name] = {group: sum(r["scores"]["strict_exact_match"] for r in items) / len(items)
                              for group, items in groups.items()}
    group_ids = sorted(group_scores["control"])
    differences = [group_scores["hard"][group] - group_scores["control"][group] for group in group_ids]
    rng = random.Random(20261010)
    replicates = sorted(sum(rng.choice(differences) for _ in differences) / len(differences)
                        for _ in range(5000))
    report["hard_minus_control"] = {
        "strict_exact_rate_difference": sum(differences) / len(differences),
        "group_bootstrap_95_percentile_interval": [replicates[124], replicates[4874]],
        "resampling_unit": "parent_group", "groups": len(group_ids), "replicates": 5000, "seed": 20261010,
        "limitation": "Conditional on these synthetic groups and one training seed; excludes training-seed uncertainty.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/v2_experiment_01")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        parser.error("Experiment directory already exists; inspect it instead of overwriting")
    lock = (ROOT / "outputs/.nameplate_v2.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another v2 runner is active")
    # Stage-2 checkpoint jobs were launched before this queue was constructed.
    # Verify they completed before acquiring the model device here.
    old_checkpoint_dirs = {
        50: ROOT / "outputs/v2_checkpoint50_val_01",
        100: ROOT / "outputs/v2_checkpoint100_val_01",
        150: ROOT / "results/nameplate_lora_val_01",
    }
    for step, folder in old_checkpoint_dirs.items():
        if not (folder / "summary.json").is_file():
            parser.error(f"Checkpoint {step} validation is not complete yet")
        if len(predictions(folder)) != 100:
            parser.error(f"Checkpoint {step} validation is incomplete")
    run_dir.mkdir(parents=True)
    script_names = ("run_nameplate_v2_experiment.py", "evaluate_nameplates_v2.py", "evaluate_nameplates.py",
                    "analyze_nameplate_predictions.py", "generate_nameplates_v2.py", "verify_nameplates_v2.py",
                    "train.py", "check_tokens.py")
    frozen_paths = [ROOT / "scripts" / name for name in script_names] + [
        ROOT / f"configs/train_nameplates_v2_{arm}_mps.yaml" for arm in ("control", "hard")]
    frozen_paths += [ROOT / "data/nameplates_v2/manifest.json", ROOT / "data/nameplates_v1/train.json"]
    data_manifest = read(ROOT / "data/nameplates_v2/manifest.json")
    frozen_paths += [ROOT / "data/nameplates_v2" / name for name in data_manifest["files_sha256"]]
    protocol = {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "arms": ["control", "hard"], "samples_per_arm": 800, "optimizer_steps": 150,
        "candidate_steps": [50, 100, 150], "batch_size": 1, "gradient_accumulation": 8,
        "selection": ["maximize v2 validation strict exact count", "minimize missing-field non-null count",
                      "minimize present-field null count", "prefer earlier checkpoint"],
        "final_test_policy": "Freeze both selected checkpoint hashes before any v2 test inference.",
        "normalization": "nameplate_normalization_v1, frozen before v2 inference",
        "uncertainty": "50 correlated groups per eval split; report counts and group-level paired bootstrap.",
        "interpretation": "Single training seed; compares the complete paired-missing-data recipe. "
                          "Synthetic performance is not real-photo performance.",
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in frozen_paths},
    }
    dump(run_dir / "protocol.json", protocol)
    state = {"status": "running", "pid": os.getpid(), "started_at": protocol["frozen_at"],
             "current_stage": None, "completed_stages": [], "failed_stage": None}

    def update():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        dump(run_dir / "status.json", state)

    def assert_frozen():
        for name, checksum in protocol["sha256"].items():
            if digest(ROOT / name) != checksum:
                raise ValueError(f"Frozen experiment file changed: {name}")

    def execute(name, command):
        assert_frozen()
        state.update(current_stage=name, command=command, stage_started_at=datetime.now(timezone.utc).isoformat())
        update()
        log = run_dir / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"START {name}", flush=True)
        with log.open("x") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            state["child_pid"] = process.pid
            update()
            try:
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        update()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {log}")
        state["completed_stages"].append(name)
        state.pop("child_pid", None)
        update()
        print(f"DONE {name}", flush=True)

    def evaluate(name, split, adapter=None, dataset="nameplates_v2", extra=()):
        output = run_dir / name
        command = [PYTHON, "-u", "scripts/evaluate_nameplates_v2.py", "--split", split,
                   "--dataset-dir", str(ROOT / "data" / dataset), "--output-dir", str(output)]
        if adapter:
            command += ["--adapter", str(adapter)]
        execute(name, command + list(extra))
        return output

    try:
        update()
        execute("verify_v1", [PYTHON, "scripts/verify_repository.py"])
        execute("verify_v2", [PYTHON, "scripts/verify_nameplates_v2.py", "--output", str(run_dir / "data_check.json")])
        old_comparison = {str(step): summarize(predictions(folder)) for step, folder in old_checkpoint_dirs.items()}
        dump(run_dir / "v1_checkpoint_comparison.json", old_comparison)
        evaluate("merged_v1_test", "test", dataset="nameplates_v1", extra=(
            "--model", str(ROOT / "outputs/nameplate_merged_01"),
            "--reference-dir", str(ROOT / "results/nameplate_lora_test_01"), "--require-match"))
        for arm in ("control", "hard"):
            execute(f"tokens_{arm}", [PYTHON, "scripts/check_tokens.py", "--config",
                    f"configs/train_nameplates_v2_{arm}_mps.yaml", "--output", str(run_dir / f"tokens_{arm}.json")])
        execute("smoke_hard", [PYTHON, "-u", "scripts/train.py", "--config", "configs/train_nameplates_v2_hard_mps.yaml",
                "--smoke", "--output-dir", str(run_dir / "smoke_hard")])
        candidates, selections = {}, {}
        for arm in ("control", "hard"):
            trained = run_dir / f"train_{arm}"
            execute(f"train_{arm}", [PYTHON, "-u", "scripts/train.py", "--config",
                    f"configs/train_nameplates_v2_{arm}_mps.yaml", "--output-dir", str(trained)])
            candidates[arm] = []
            for step in (50, 100, 150):
                adapter = trained / f"checkpoint-{step}"
                folder = evaluate(f"{arm}_step{step}_val", "val", adapter)
                candidates[arm].append({"step": step, "adapter": str(adapter.relative_to(ROOT)),
                                        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
                                        "metrics": summarize(predictions(folder))})
            selections[arm] = min(candidates[arm], key=selection_key)
            dump(run_dir / "validation_candidates.json", candidates)
        # This immutable selection artifact exists before any final-test model call.
        dump(run_dir / "selection.json", {"frozen_at": datetime.now(timezone.utc).isoformat(),
                                          "protocol_sha256": digest(run_dir / "protocol.json"),
                                          "selected": selections})
        adapters = {"base": None, "v1_adapter": ROOT / "artifacts/nameplate_lora_01",
                    **{arm: ROOT / selected["adapter"] for arm, selected in selections.items()}}
        final = {"protocol": protocol, "selection": selections, "test": {}, "v1_validation_regression": {}}
        for name, adapter in adapters.items():
            if name in selections and digest(adapter / "adapter_model.safetensors") != selections[name]["adapter_sha256"]:
                raise ValueError("Selected adapter changed after selection was frozen")
            folder = evaluate(f"{name}_test", "test", adapter)
            final["test"][name] = summarize(predictions(folder))
            if name != "base":
                execute(f"analysis_{name}_test", [PYTHON, "scripts/analyze_nameplate_predictions.py", "--split", "test",
                        "--dataset-dir", str(ROOT / "data/nameplates_v2"), "--base-dir", str(run_dir / "base_test"),
                        "--lora-dir", str(folder), "--output", str(run_dir / f"analysis_{name}_test.json")])
        for arm in ("control", "hard"):
            folder = evaluate(f"{arm}_v1_val", "val", ROOT / selections[arm]["adapter"], dataset="nameplates_v1")
            final["v1_validation_regression"][arm] = summarize(predictions(folder))
        final["grouped_test"] = grouped_test_report(run_dir)
        dump(run_dir / "final_summary.json", final)
        state.update(status="completed", current_stage=None, completed_at=datetime.now(timezone.utc).isoformat())
        update()
        print(f"COMPLETE {run_dir / 'final_summary.json'}", flush=True)
    except BaseException as error:
        state.update(status="failed", failed_stage=state.get("current_stage"), error=str(error))
        update()
        raise


if __name__ == "__main__":
    main()
