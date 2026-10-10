"""Guard against score inflation and conflating missing values with missing keys."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_nameplate_predictions import (
    compare_runs, diagnostic_object, normalize_object, score_row, summarize,
)


def row(raw, expected):
    return score_row({"index": 1, "images": ["a.png"], "prompt": "extract",
                      "raw_response": raw, "expected": expected})


class PredictionAnalysisTests(unittest.TestCase):
    def test_normalization_keeps_identifiers_and_is_idempotent(self):
        original = {"型号": " AC 12 ", "电压": " AC 220 V ", "功率": "N/A"}
        saved = copy.deepcopy(original)
        normalized = normalize_object(original)
        self.assertEqual(normalized, {"型号": " AC 12 ", "电压": "220V", "功率": None})
        self.assertEqual(normalize_object(normalized), normalized)
        self.assertEqual(original, saved)

    def test_no_numeric_or_unit_repair(self):
        self.assertEqual(normalize_object({"电压": "22O V", "功率": "1.5kW"}),
                         {"电压": "22OV", "功率": "1.5kW"})
        self.assertEqual(normalize_object({"电压": "220"}), {"电压": "220"})

    def test_rejects_invalid_json_and_prose(self):
        for raw in ('{"型号": "A", "型号": "B"}', '{"功率": NaN}',
                    'answer: {"型号": "A"}', '```json\n{}\n``` extra', '[]'):
            with self.subTest(raw=raw):
                self.assertIsNone(diagnostic_object(raw))

    def test_fence_is_diagnostic_only(self):
        expected = {"型号": "A", "电压": "220V", "功率": None}
        item = row('```json\n{"型号":"A","电压":"AC 220 V","功率":"null"}\n```', expected)
        self.assertFalse(item["scores"]["strict_exact_match"])
        self.assertFalse(item["scores"]["diagnostic_exact_match"])
        self.assertTrue(item["normalized_exact_match"])

    def test_missing_keys_and_extra_keys_are_not_repaired(self):
        expected = {"型号": "A", "电压": None, "功率": None}
        for raw in ('{"型号":"A","功率":null}',
                    '{"型号":"A","电压":null,"功率":null,"extra":1}'):
            with self.subTest(raw=raw):
                self.assertFalse(row(raw, expected)["normalized_exact_match"])

    def test_null_precision_recall_cover_both_error_directions(self):
        items = [row('{"型号":"A","电压":"220V","功率":null}',
                     {"型号": "A", "电压": None, "功率": "200W"}),
                 row('{"型号":"B","电压":null,"功率":"50W"}',
                     {"型号": "B", "电压": None, "功率": "50W"})]
        metrics = summarize(items)["null_metrics_normalized"]
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["spurious_value_rate"], 0.5)
        self.assertEqual(metrics["false_null_rate"], 0.25)

    def test_invalid_response_stays_in_denominator(self):
        item = row("not json", {"型号": "A", "电压": None, "功率": "200W"})
        metrics = summarize([item])
        self.assertEqual(metrics["rates"]["normalized_exact_match"], 0)
        self.assertEqual(metrics["null_metrics_normalized"]["recall"], 0)
        self.assertIsNone(metrics["null_metrics_normalized"]["precision"])
        self.assertEqual(metrics["null_metrics_normalized"]["spurious_value_count"], 0)

    def test_pairing_rejects_different_samples(self):
        expected = {"型号": "A", "电压": None, "功率": None}
        a, b = row('{}', expected), row('{}', expected)
        b["images"] = ["b.png"]
        with self.assertRaisesRegex(ValueError, "not paired"):
            compare_runs([a], [b], {})

    def test_reports_regressions_and_slice_denominators(self):
        expected = {"型号": "A", "电压": None, "功率": "200W"}
        a = row('{"型号":"A","电压":null,"功率":"200 W"}', expected)
        b = row('{"型号":"A","电压":null,"功率":null}', expected)
        metadata = {"a.png": {"answer": expected, "missing_field": "电压",
                             "missing_mode": "blank", "template": "T01", "font": "font.ttf"}}
        report = compare_runs([a], [b], metadata)
        self.assertEqual(report["paired_normalized_comparison"]["regressed_count"], 1)
        sliced = report["slices"]["missing_field_and_mode"]["电压:blank"]
        self.assertEqual(sliced["lora"]["sample_count"], 1)
        self.assertEqual(sliced["lora"]["null_metrics_normalized"]["fp"], 1)


if __name__ == "__main__":
    unittest.main()
