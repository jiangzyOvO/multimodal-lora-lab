"""Reproduce the v2 training comparison without any historical outputs.

Run --dry-run to inspect the plan, --check-only to check prerequisites, or
--smoke for one optimizer update and two validation generations. With no mode
flag, train both arms for 150 updates and evaluate the complete datasets.
"""
import argparse
import fcntl
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from evaluate_nameplates_v2 import artifact_hashes, sha256
from publish_nameplate_v2 import verify_evaluation
from run_nameplate_v2_experiment import dump, grouped_test_report, read, selection_key

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control", "hard")
STEPS = (50, 100, 150)
FRAMEWORK_REVISION = "7af909522a951e3ad9f022ea6f88b6755257eaa5"
MODEL = Path("models/Qwen2-VL-2B-Instruct")
SCRIPTS = (
    "reproduce_nameplates_v2.py", "train.py", "check_tokens.py",
    "evaluate_nameplates.py", "evaluate_nameplates_v2.py",
    "analyze_nameplate_predictions.py", "run_nameplate_v2_experiment.py",
    "publish_nameplate_v2.py", "verify_repository.py", "verify_nameplate_export.py",
    "verify_nameplates_v2.py", "generate_nameplates.py", "generate_nameplates_v2.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_run_dir(value, root=ROOT):
    path = (root / value).resolve()
    output = (root / "outputs").resolve()
    if path == output or not path.is_relative_to(output):
        raise ValueError("--run-dir must be a new subdirectory of outputs/")
    if path.exists():
        raise ValueError(f"Run directory already exists; choose a new name: {path}")
    return path


def build_plan(run_dir, smoke=False, root=ROOT):
    """Build commands without probing the environment or creating directories."""
    python = str(root / ".venv/bin/python")
    stages = []

    def command(name, script, *args):
        stages.append({"name": name, "command": [python, "-u", f"scripts/{script}", *map(str, args)]})

    def evaluate(name, split, adapter=None, dataset="nameplates_v2", limit=None):
        args = ["--split", split, "--dataset-dir", root / "data" / dataset,
                "--output-dir", run_dir / name]
        if adapter:
            args += ["--adapter", adapter]
        if limit:
            args += ["--limit", limit]
        command(name, "evaluate_nameplates_v2.py", *args)

    command("verify_v1", "verify_repository.py")
    command("verify_v2", "verify_nameplates_v2.py", "--output", run_dir / "data_check.json")
    for arm in ARMS:
        command(f"tokens_{arm}", "check_tokens.py", "--config", run_dir / f"configs/{arm}.json",
                "--output", run_dir / f"tokens_{arm}.json")
    command("smoke_hard", "train.py", "--config", run_dir / "configs/hard.json",
            "--smoke", "--output-dir", run_dir / "smoke_hard")
    if smoke:
        evaluate("smoke_val", "val", run_dir / "smoke_hard", limit=2)
        stages.append({"name": "smoke_report", "action": "smoke_report"})
        return stages
    for arm in ARMS:
        command(f"train_{arm}", "train.py", "--config", run_dir / f"configs/{arm}.json",
                "--output-dir", run_dir / f"train_{arm}")
        for step in STEPS:
            evaluate(f"{arm}_step{step}_val", "val", run_dir / f"train_{arm}/checkpoint-{step}")
    stages.append({"name": "freeze_selection", "action": "freeze_selection"})
    for name, adapter in (
        ("base", None), ("v1_adapter", root / "artifacts/nameplate_lora_01"),
        ("control", "{selected_control}"), ("hard", "{selected_hard}"),
    ):
        evaluate(f"{name}_test", "test", adapter)
    for arm in ARMS:
        evaluate(f"{arm}_v1_val", "val", "{selected_" + arm + "}", dataset="nameplates_v1")
    stages.append({"name": "final_report", "action": "final_report"})
    return stages


def required_files(root=ROOT):
    paths = [root / "scripts" / name for name in SCRIPTS]
    paths += [root / "requirements.lock.txt", root / "data/dataset_info.json",
              root / "artifacts/nameplate_lora_01/adapter_config.json",
              root / "artifacts/nameplate_lora_01/adapter_model.safetensors",
              root / "results/v2_experiment_01/base_test/run.json"]
    paths += [root / f"configs/train_nameplates_v2_{arm}_mps.yaml" for arm in ARMS]
    for version in ("v1", "v2"):
        directory = root / f"data/nameplates_{version}"
        paths += [directory / "manifest.json", directory / "metadata.jsonl"]
        manifest = directory / "manifest.json"
        if manifest.is_file():
            paths += [directory / name for name in read(manifest)["files_sha256"]]
        metadata = directory / "metadata.jsonl"
        if metadata.is_file():
            paths += [root / "data" / json.loads(line)["image"]
                      for line in metadata.read_text().splitlines() if line]
    return sorted(set(paths))


def preflight(root=ROOT):
    """Collect prerequisites; no model loading, downloads, or output writes."""
    errors = []
    paths = required_files(root)
    missing = [str(p.relative_to(root)) for p in paths if not p.is_file()]
    if missing:
        errors.append("Missing repository files: " + ", ".join(missing[:12])
                      + (f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""))
    python = root / ".venv/bin/python"
    if not python.is_file():
        errors.append("Missing .venv/bin/python; run bash scripts/setup.sh")
    framework = root / "LLaMA-Factory"
    if not (framework / ".git").exists():
        errors.append("Missing LLaMA-Factory checkout; run bash scripts/setup.sh")
    else:
        revision = subprocess.run(["git", "-C", str(framework), "rev-parse", "HEAD"],
                                  capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(framework), "diff", "--quiet", "HEAD"])
        if revision.returncode or revision.stdout.strip() != FRAMEWORK_REVISION or dirty.returncode:
            errors.append("LLaMA-Factory must match the clean pinned revision from scripts/setup.sh")
    environment = {}
    if python.is_file():
        probe = (
            "import json,sys,platform,torch; from importlib.metadata import version; "
            "import llamafactory, transformers, peft, accelerate, torchvision, PIL, yaml; "
            "print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(), "
            "'mps_available': torch.backends.mps.is_available(), "
            "'versions': {p:version(p) for p in ['torch','torchvision','transformers',"
            "'accelerate','peft','pillow','pyyaml','numpy','llamafactory']}}))"
        )
        result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True)
        if result.returncode:
            errors.append("Dependency import failed; rerun scripts/setup.sh: " + result.stderr[-1000:])
        else:
            environment = json.loads(result.stdout.strip().splitlines()[-1])
            if not environment["mps_available"]:
                errors.append("MPS unavailable; this reproduction uses the recorded Apple/MPS configuration")
            if environment["python"] != "3.11.17":
                errors.append("Expected Python 3.11.17; recreate the environment with scripts/setup.sh")
            if environment["versions"]["llamafactory"] != "0.9.5":
                errors.append("Expected installed llamafactory 0.9.5; rerun scripts/setup.sh")
            lock = root / "requirements.lock.txt"
            pins = dict(line.split("==", 1) for line in lock.read_text().splitlines()
                        if "==" in line and not line.startswith("#")) if lock.is_file() else {}
            for package, version in environment["versions"].items():
                if package in pins and version != pins[package]:
                    errors.append(f"{package}: expected {pins[package]}, found {version}")
    reference = root / "results/v2_experiment_01/base_test/run.json"
    model_hashes = {}
    if reference.is_file():
        expected = read(reference)["model_sha256"]
        absent = [name for name in [*expected, "merges.txt", "LICENSE"]
                  if not (root / MODEL / name).is_file()]
        if absent:
            errors.append("Missing base model files: " + ", ".join(absent)
                          + "; run .venv/bin/python scripts/download_model.py")
        else:
            model_hashes = artifact_hashes(root / MODEL)
            if model_hashes != expected:
                errors.append("Base model fingerprint differs from the published fixed snapshot")
    return {"passed": not errors, "errors": errors, "environment": environment,
            "model_sha256": model_hashes}


def check_audit(path, steps):
    audit = read(path / "training_audit.json")
    if not (audit["completed_steps"] == steps and audit["changed_tensors"] > 0
            and audit["all_adapter_values_finite"]
            and audit["all_trainable_are_language_lora"]
            and audit["trainable_parameters"] == 544768):
        raise ValueError(f"Training audit failed: {path}")
    return audit


def freeze_selection(run_dir, protocol, root=ROOT):
    candidates, selected = {}, {}
    for arm in ARMS:
        check_audit(run_dir / f"train_{arm}", 150)
        candidates[arm] = []
        for step in STEPS:
            folder = run_dir / f"{arm}_step{step}_val"
            metrics = verify_evaluation(folder, root / "data/nameplates_v2/val.json")
            adapter = run_dir / f"train_{arm}/checkpoint-{step}"
            hashes = artifact_hashes(adapter)
            run = read(folder / "run.json")
            if run["adapter_sha256"] != hashes or run["model_sha256"] != protocol["model_sha256"]:
                raise ValueError(f"Validation used different weights: {folder}")
            candidates[arm].append({"step": step, "adapter": str(adapter.relative_to(root)),
                                    "adapter_sha256": hashes, "metrics": metrics})
        selected[arm] = min(candidates[arm], key=selection_key)
    dump(run_dir / "validation_candidates.json", candidates)
    selection = {"frozen_at": now(), "protocol_sha256": sha256(run_dir / "protocol.json"),
                 "selected": selected}
    dump(run_dir / "selection.json", selection)
    return selected


def selected_arguments(command, selected, root=ROOT):
    replacements = {}
    for arm, candidate in selected.items():
        path = root / candidate["adapter"]
        if artifact_hashes(path) != candidate["adapter_sha256"]:
            raise ValueError(f"Selected adapter changed after selection: {arm}")
        replacements["{selected_" + arm + "}"] = str(path)
    return [replacements.get(argument, argument) for argument in command]


def finalize(run_dir, protocol, root=ROOT):
    selection = read(run_dir / "selection.json")
    if selection["protocol_sha256"] != sha256(run_dir / "protocol.json"):
        raise ValueError("Selection protocol differs")
    final = {"protocol": protocol, "selection": selection["selected"],
             "test": {}, "v1_validation_regression": {}}
    frozen_at = datetime.fromisoformat(selection["frozen_at"])
    for name in ("base", "v1_adapter", *ARMS):
        folder = run_dir / f"{name}_test"
        run = read(folder / "run.json")
        expected = (selection["selected"][name]["adapter_sha256"] if name in ARMS else
                    artifact_hashes(root / "artifacts/nameplate_lora_01") if name == "v1_adapter" else None)
        if (run["model_sha256"] != protocol["model_sha256"] or run["adapter_sha256"] != expected
                or datetime.fromisoformat(run["started_at"]) < frozen_at):
            raise ValueError(f"Test weights or selection order differ: {name}")
        final["test"][name] = verify_evaluation(folder, root / "data/nameplates_v2/test.json")
    for arm in ARMS:
        folder = run_dir / f"{arm}_v1_val"
        run = read(folder / "run.json")
        if (run["adapter_sha256"] != selection["selected"][arm]["adapter_sha256"]
                or run["model_sha256"] != protocol["model_sha256"]):
            raise ValueError(f"Regression used different weights: {arm}")
        final["v1_validation_regression"][arm] = verify_evaluation(
            folder, root / "data/nameplates_v1/val.json")
    final["grouped_test"] = grouped_test_report(run_dir)
    dump(run_dir / "final_summary.json", final)
    lines = ["# v2 独立复现结果", "", "两组从同一基础模型训练，seed=42，各 800 条、150 次更新。",
             "checkpoint 仅用验证集选择，并在最终测试前固定权重。", "",
             "|模型|严格全对|规则处理后全对|", "|---|---:|---:|"]
    for name, metrics in final["test"].items():
        counts, n = metrics["counts"], metrics["sample_count"]
        lines.append(f"|{name}|{counts['strict_exact_match']}/{n}|{counts['normalized_exact_match']}/{n}|")
    delta = final["grouped_test"]["hard_minus_control"]
    lo, hi = delta["group_bootstrap_95_percentile_interval"]
    lines += ["", f"困难减普通：{100 * delta['strict_exact_rate_difference']:+.1f} 个百分点；"
              f"父组配对 bootstrap 95% 区间 [{100 * lo:+.1f}, {100 * hi:+.1f}]。",
              "", "该结果限于本批合成父组和一个训练种子，不含训练随机性或真实照片泛化证据。",
              "本次只复现 v2 对照；不包含历史 v1 checkpoint 对比和合并模型验证。",
              "协议、选择、环境、指纹及运行阶段见 protocol.json、selection.json、preflight.json、status.json。",
              "原始预测和逐组统计见各评估目录及 final_summary.json。", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_workflow(run_dir, plan, protocol, execute, root=ROOT):
    """Keep ordering testable without allocating a model."""
    selected = {}
    for stage in plan:
        if "command" in stage:
            execute(stage["name"], selected_arguments(stage["command"], selected, root))
        elif stage["action"] == "freeze_selection":
            execute(stage["name"], lambda: selected.update(freeze_selection(run_dir, protocol, root)))
        elif stage["action"] == "final_report":
            execute(stage["name"], lambda: finalize(run_dir, protocol, root))
        else:
            def smoke_report():
                audit = check_audit(run_dir / "smoke_hard", 1)
                run = read(run_dir / "smoke_val/run.json")
                if (run["status"] != "completed" or run["sample_count"] != 2
                        or run["model_sha256"] != protocol["model_sha256"]
                        or run["adapter_sha256"] != artifact_hashes(run_dir / "smoke_hard")):
                    raise ValueError("Smoke inference incomplete or weights differ")
                dump(run_dir / "smoke_summary.json", {
                    "mode": "smoke", "completed_steps": audit["completed_steps"],
                    "validation_samples": 2, "full_experiment_completed": False,
                    "note": "Only a training/inference smoke test; not an accuracy estimate.",
                })
            execute(stage["name"], smoke_report)


def run_experiment(run_dir, plan, prerequisites, smoke=False):
    import yaml

    # Share the historical runner's lock so the two queues cannot overlap.
    (ROOT / "outputs").mkdir(exist_ok=True)
    with (ROOT / "outputs/.nameplate_v2.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another v2 queue is active") from error
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {"mode": "smoke" if smoke else "full", "status": "running", "pid": os.getpid(),
                 "started_at": now(), "current_stage": "prepare", "completed_stages": []}

        def update():
            state["updated_at"] = now()
            dump(run_dir / "status.json", state)

        try:
            update()
            dump(run_dir / "preflight.json", prerequisites)
            dump(run_dir / "plan.json", plan)
            for arm in ARMS:
                config = yaml.safe_load((ROOT / f"configs/train_nameplates_v2_{arm}_mps.yaml").read_text())
                config["output_dir"] = str(run_dir / f"train_{arm}")
                dump(run_dir / f"configs/{arm}.json", config)
            frozen = required_files() + list((run_dir / "configs").glob("*.json"))
            protocol = {
                "entrypoint": "reproduce_nameplates_v2", "mode": state["mode"], "frozen_at": now(),
                "seed": 42, "arms": list(ARMS), "samples_per_arm": 800,
                "optimizer_steps": 1 if smoke else 150, "candidate_steps": [] if smoke else list(STEPS),
                "batch_size": 1, "gradient_accumulation": 8,
                "selection": ["maximize validation strict exact count", "minimize spurious values",
                              "minimize false nulls", "prefer earlier checkpoint"],
                "normalization": "nameplate_normalization_v1",
                "model_sha256": prerequisites["model_sha256"],
                "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in frozen},
                "scope": "v2 comparison only; historical v1 checkpoint/export audits are independent.",
            }
            dump(run_dir / "protocol.json", protocol)
            model_stats = {name: ((ROOT / MODEL / name).stat().st_size,
                                  (ROOT / MODEL / name).stat().st_mtime_ns)
                           for name in protocol["model_sha256"]}

            def assert_frozen():
                for name, expected in protocol["sha256"].items():
                    if sha256(ROOT / name) != expected:
                        raise ValueError(f"Frozen input changed: {name}")
                for name, expected in model_stats.items():
                    stat = (ROOT / MODEL / name).stat()
                    if (stat.st_size, stat.st_mtime_ns) != expected:
                        raise ValueError(f"Base model changed during the run: {name}")

            def execute(name, operation):
                assert_frozen()
                state.update(current_stage=name, stage_started_at=now())
                state["command"] = operation if isinstance(operation, list) else None
                update()
                print(f"START {name}", flush=True)
                if callable(operation):
                    operation()
                else:
                    log = run_dir / f"logs/{name}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("x") as handle:
                        child = subprocess.Popen(operation, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                        state["child_pid"] = child.pid
                        update()
                        try:
                            while True:
                                try:
                                    code = child.wait(timeout=30)
                                    break
                                except subprocess.TimeoutExpired:
                                    update()
                        except BaseException:
                            child.terminate()
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            raise
                        finally:
                            state.pop("child_pid", None)
                    if code:
                        raise RuntimeError(f"{name} exited {code}; inspect {log}")
                assert_frozen()
                state["completed_stages"].append(name)
                update()
                print(f"DONE {name}", flush=True)

            run_workflow(run_dir, plan, protocol, execute)
            state.update(status="completed", current_stage=None, completed_at=now())
            update()
            print(f"COMPLETE mode={state['mode']} {run_dir}", flush=True)
        except BaseException as error:
            state.update(status="failed", failed_stage=state.get("current_stage"), error=str(error))
            update()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/v2_reproduction_01"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Print the plan only; no environment checks or writes.")
    group.add_argument("--check-only", action="store_true", help="Check prerequisites only; no training or writes.")
    parser.add_argument("--smoke", action="store_true", help="Only one update and two validation generations.")
    args = parser.parse_args()
    try:
        run_dir = resolve_run_dir(args.run_dir)
        plan = build_plan(run_dir, args.smoke)
        if args.dry_run:
            print("Plan only; not executed. Runtime configs will be created under " + str(run_dir / "configs"))
            for stage in plan:
                print(stage["name"] + ": " + (shlex.join(stage["command"])
                                             if "command" in stage else stage["action"]))
            return
        prerequisites = preflight()
        print(json.dumps(prerequisites, ensure_ascii=False, indent=2), flush=True)
        if not prerequisites["passed"]:
            raise SystemExit(1)
        if not args.check_only:
            run_experiment(run_dir, plan, prerequisites, args.smoke)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
