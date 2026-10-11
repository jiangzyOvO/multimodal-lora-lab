import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reproduce_nameplates_v2 as reproduction


class ReproductionEntrypointTests(unittest.TestCase):
    def test_clean_checkout_plan_has_no_historical_output_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = reproduction.resolve_run_dir(Path("outputs/fresh"), root)
            plan = reproduction.build_plan(run, root=root)
            self.assertFalse((root / "outputs").exists())
            for stage in plan:
                for argument in stage.get("command", []):
                    if str(root / "outputs") in argument:
                        self.assertTrue(Path(argument).is_relative_to(run), argument)
            self.assertNotIn("merged_v1_test", [stage["name"] for stage in plan])
            self.assertNotIn("v2_checkpoint50_val_01", json.dumps(plan))
            self.assertNotIn("v2_checkpoint100_val_01", json.dumps(plan))

    def test_selection_precedes_every_final_test_and_smoke_does_not_use_test(self):
        run = Path("/example/outputs/new")
        plan = reproduction.build_plan(run, root=Path("/example"))
        names = [stage["name"] for stage in plan]
        selection_index = names.index("freeze_selection")
        for name in ("base_test", "v1_adapter_test", "control_test", "hard_test"):
            self.assertGreater(names.index(name), selection_index)
        for arm in reproduction.ARMS:
            for step in reproduction.STEPS:
                self.assertLess(names.index(f"{arm}_step{step}_val"), selection_index)
        smoke = reproduction.build_plan(run, smoke=True, root=Path("/example"))
        self.assertFalse(any(s["name"].endswith("_test") for s in smoke))
        self.assertFalse(any(s["name"].startswith("train_") for s in smoke))
        self.assertEqual(smoke[-1]["action"], "smoke_report")

    def test_existing_and_outside_output_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs/existing").mkdir(parents=True)
            for name in ("outputs", "outputs/existing", "../outside", "results/new"):
                with self.assertRaises(ValueError):
                    reproduction.resolve_run_dir(Path(name), root)

    def test_missing_prerequisites_are_reported_together_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = reproduction.preflight(root)
            self.assertFalse(result["passed"])
            self.assertGreaterEqual(len(result["errors"]), 3)
            self.assertFalse((root / "outputs").exists())

    def test_dry_run_works_without_environment_model_or_outputs(self):
        # Execute the CLI from a clean directory with just its stdlib-only script dependencies.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            for name in reproduction.SCRIPTS:
                (scripts / name).write_bytes((reproduction.ROOT / "scripts" / name).read_bytes())
            result = subprocess.run(
                [sys.executable, str(scripts / "reproduce_nameplates_v2.py"), "--dry-run"],
                cwd="/", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("freeze_selection", result.stdout)
            self.assertIn("{selected_control}", result.stdout)
            self.assertFalse((root / "outputs").exists())

    def test_modified_selected_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "outputs/train/checkpoint-50"
            adapter.mkdir(parents=True)
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"original")
            selected = {"control": {"adapter": str(adapter.relative_to(root)),
                                    "adapter_sha256": reproduction.artifact_hashes(adapter)}}
            self.assertEqual(reproduction.selected_arguments(["--adapter", "{selected_control}"], selected, root),
                             ["--adapter", str(adapter)])
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after selection"):
                reproduction.selected_arguments(["{selected_control}"], selected, root)

    def test_workflow_freezes_validation_winner_before_test_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root, executed = Path(directory), []
            run = root / "outputs/fresh"
            run.mkdir(parents=True)
            protocol = {"model_sha256": {"base": "hash"}}
            reproduction.dump(run / "protocol.json", protocol)
            for arm in reproduction.ARMS:
                for step in reproduction.STEPS:
                    path = run / f"train_{arm}/checkpoint-{step}"
                    path.mkdir(parents=True)
                    (path / "adapter_model.safetensors").write_bytes(f"{arm}-{step}".encode())
                    folder = run / f"{arm}_step{step}_val"
                    reproduction.dump(folder / "run.json", {
                        "adapter_sha256": reproduction.artifact_hashes(path),
                        "model_sha256": protocol["model_sha256"],
                    })
            def metrics(folder, dataset):
                step = int(folder.name.split("_step")[1].split("_")[0])
                return {"counts": {"strict_exact_match": 198 if step == 100 else 190},
                        "null_metrics_fence_only": {"spurious_value_count": 1, "fp": 0}}

            def execute(name, operation):
                executed.append(name)
                if callable(operation):
                    operation()
                elif name.endswith("_test"):
                    self.assertTrue((run / "selection.json").is_file())
                    if name in ("control_test", "hard_test"):
                        self.assertTrue(operation[operation.index("--adapter") + 1].endswith("checkpoint-100"))

            with patch.object(reproduction, "check_audit"), \
                    patch.object(reproduction, "verify_evaluation", side_effect=metrics), \
                    patch.object(reproduction, "finalize") as finalize:
                reproduction.run_workflow(run, reproduction.build_plan(run, root=root),
                                          protocol, execute, root)
                finalize.assert_called_once()
            self.assertEqual(reproduction.read(run / "selection.json")["selected"]["hard"]["step"], 100)
            self.assertLess(executed.index("freeze_selection"), executed.index("base_test"))

    def test_failed_stage_stops_workflow_before_selection_and_tests(self):
        run = Path("/example/outputs/new")
        executed = []
        def execute(name, operation):
            executed.append(name)
            if name == "train_control":
                raise RuntimeError("training failed")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            reproduction.run_workflow(run, reproduction.build_plan(run), {}, execute)
        self.assertNotIn("freeze_selection", executed)
        self.assertNotIn("base_test", executed)

    def test_final_report_rejects_test_started_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "outputs/fresh"
            protocol = {"model_sha256": {"base": "hash"}}
            reproduction.dump(run / "protocol.json", protocol)
            reproduction.dump(run / "selection.json", {
                "frozen_at": "2026-10-11T12:00:00+00:00",
                "protocol_sha256": reproduction.sha256(run / "protocol.json"), "selected": {},
            })
            reproduction.dump(run / "base_test/run.json", {
                "model_sha256": protocol["model_sha256"], "adapter_sha256": None,
                "started_at": "2026-10-11T11:59:00+00:00",
            })
            with self.assertRaisesRegex(ValueError, "selection order"):
                reproduction.finalize(run, protocol, root)

    def test_final_report_recomputes_published_predictions(self):
        # Exercise report assembly on known raw outputs, without claiming a new model run.
        source = reproduction.ROOT / "results/v2_experiment_01"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "outputs/replay"
            run.mkdir(parents=True)
            for version, split in (("v2", "test"), ("v1", "val")):
                target = root / f"data/nameplates_{version}/{split}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((reproduction.ROOT / f"data/nameplates_{version}/{split}.json").read_bytes())
            artifact = root / "artifacts/nameplate_lora_01"
            shutil.copytree(reproduction.ROOT / "artifacts/nameplate_lora_01", artifact)
            for name in ("base_test", "v1_adapter_test", "control_test", "hard_test",
                         "control_v1_val", "hard_v1_val"):
                shutil.copytree(source / name, run / name)
            protocol = {"model_sha256": reproduction.read(run / "base_test/run.json")["model_sha256"]}
            reproduction.dump(run / "protocol.json", protocol)
            reproduction.dump(run / "selection.json", {
                "frozen_at": "2020-01-01T00:00:00+00:00",
                "protocol_sha256": reproduction.sha256(run / "protocol.json"),
                "selected": {arm: {"adapter_sha256": reproduction.read(
                    run / f"{arm}_test/run.json")["adapter_sha256"]} for arm in reproduction.ARMS},
            })
            reproduction.finalize(run, protocol, root)
            final = reproduction.read(run / "final_summary.json")
            self.assertEqual(final["test"]["hard"]["counts"]["strict_exact_match"], 198)
            self.assertAlmostEqual(final["grouped_test"]["hard_minus_control"]["strict_exact_rate_difference"], .025)
            self.assertIn("198/200", (run / "REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
