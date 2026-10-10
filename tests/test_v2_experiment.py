import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_nameplates_v2 import compare_predictions
from generate_nameplates_v2 import missing_variant
from verify_nameplates_v2 import answer_from_annotations
from run_nameplate_v2_experiment import selection_key


class V2ExperimentTests(unittest.TestCase):
    def test_selection_uses_validation_then_missing_errors_then_earlier_step(self):
        def candidate(step, correct, spurious, false_null):
            return {"step": step, "metrics": {"counts": {"strict_exact_match": correct},
                    "null_metrics_fence_only": {"spurious_value_count": spurious, "fp": false_null}}}
        candidates = [candidate(50, 180, 4, 0), candidate(100, 181, 3, 1),
                      candidate(150, 181, 3, 1)]
        self.assertEqual(min(candidates, key=selection_key)["step"], 100)
        candidates.append(candidate(150, 181, 2, 3))
        self.assertEqual(min(candidates, key=selection_key)["step"], 150)

    def test_export_comparison_detects_semantic_and_text_differences(self):
        reference = {"index": 1, "images": ["a.png"], "prompt": "read", "expected": {},
                     "raw_response": "{}", "finish_reason": "stop", "scores": {"correct": True}}
        generated = copy.deepcopy(reference)
        self.assertTrue(compare_predictions([generated], [reference])["all_checks_passed"])
        generated["raw_response"] = " {} "
        comparison = compare_predictions([generated], [reference])
        self.assertFalse(comparison["all_checks_passed"])
        self.assertEqual(comparison["scores_match"], 1)
        generated["expected"] = {"different": None}
        with self.assertRaises(ValueError):
            compare_predictions([generated], [reference])

    def test_export_comparison_rejects_partial_run(self):
        with self.assertRaises(ValueError):
            compare_predictions([], [])
        with self.assertRaises(ValueError):
            compare_predictions([{}], [])

    def test_counterfactual_changes_only_target(self):
        base = {"answer": {"型号": "A", "电压": "12V", "功率": "18W"},
                "display_values": {"型号": "A", "电压": "DC 12 V", "功率": "18 W"},
                "field_order": ["功率", "型号", "电压"], "render_seed": 42}
        original = copy.deepcopy(base)
        variant = missing_variant(base, "电压", "blank")
        self.assertIsNone(variant["answer"]["电压"])
        self.assertEqual(variant["display_values"]["电压"], "")
        self.assertEqual(variant["answer"]["功率"], "18W")
        self.assertEqual(variant["field_order"], base["field_order"])
        self.assertEqual(base, original)

    def test_label_verification_uses_rendered_text_not_reference(self):
        spec = {"answer": {"型号": "wrong", "电压": "999V", "功率": "999W"},
                "rendered_text": [
                    {"kind": "value", "field": "型号", "text": "A-120"},
                    {"kind": "value", "field": "电压", "text": "AC 220 V"},
                    {"kind": "value", "field": "功率", "text": "N/A"}]}
        self.assertEqual(answer_from_annotations(spec), {"型号": "A-120", "电压": "220V", "功率": None})
        spec["rendered_text"].append({"kind": "value", "field": "电压", "text": "380 V"})
        with self.assertRaises(ValueError):
            answer_from_annotations(spec)


if __name__ == "__main__":
    unittest.main()
