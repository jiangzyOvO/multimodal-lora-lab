import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_nameplates import score_response
from publish_nameplate_v2 import validate_run, verify_prediction_records


class PublicationTests(unittest.TestCase):
    def fixture(self):
        expected = {"型号": "A", "电压": None, "功率": "200W"}
        raw = json.dumps(expected)
        sample = {"images": ["a.png"], "messages": [
            {"role": "user", "content": "read"}, {"role": "assistant", "content": raw}]}
        row = {"index": 1, "images": ["a.png"], "prompt": "read", "expected": expected,
               "raw_response": raw, "scores": score_response(raw, expected)}
        return sample, row

    def test_running_experiment_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "status.json").write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError, "not complete"):
                validate_run(path)

    def test_scores_are_recomputed_from_raw_answers(self):
        sample, row = self.fixture()
        self.assertEqual(verify_prediction_records([row], [sample])["counts"]["strict_exact_match"], 1)
        row["raw_response"] = '{"型号":"A","电压":"220V","功率":"200W"}'
        with self.assertRaisesRegex(ValueError, "recomputation"):
            verify_prediction_records([row], [sample])

    def test_reference_cannot_be_changed_to_match_prediction(self):
        sample, row = self.fixture()
        row = copy.deepcopy(row)
        row["expected"]["电压"] = "220V"
        with self.assertRaisesRegex(ValueError, "Reference"):
            verify_prediction_records([row], [sample])

    def test_missing_or_misaligned_samples_are_rejected(self):
        sample, row = self.fixture()
        with self.assertRaisesRegex(ValueError, "count"):
            verify_prediction_records([], [sample])
        row["images"] = ["b.png"]
        with self.assertRaisesRegex(ValueError, "alignment"):
            verify_prediction_records([row], [sample])


if __name__ == "__main__":
    unittest.main()
