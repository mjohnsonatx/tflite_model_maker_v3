import os
import sys
import types
import unittest


def _install_import_stubs():
    cv2_module = sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    cv2_module.FONT_HERSHEY_SIMPLEX = 0
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    pil_module = sys.modules.setdefault("PIL", types.ModuleType("PIL"))
    pil_module.__path__ = []
    sys.modules.setdefault("PIL.Image", types.ModuleType("PIL.Image"))
    sys.modules.setdefault("PIL.ImageTk", types.ModuleType("PIL.ImageTk"))


_install_import_stubs()

from InferenceTab import Detection, GroundTruth, InferenceTab


class InferenceTabTests(unittest.TestCase):
    def test_compare_detections_matches_by_label_and_iou(self):
        ground_truths = [
            GroundTruth("dumbbell", (10, 10, 110, 110)),
            GroundTruth("dumbbell", (200, 200, 260, 260)),
        ]
        predictions = [
            Detection("dumbbell", 0.95, (12, 12, 108, 108)),
            Detection("dumbbell", 0.80, (300, 300, 360, 360)),
            Detection("other", 0.90, (200, 200, 260, 260)),
        ]

        comparison = InferenceTab.compare_detections(ground_truths, predictions, 0.5)

        self.assertEqual(len(comparison["matches"]), 1)
        self.assertEqual(comparison["missed"], [1])
        self.assertEqual(set(comparison["false_positives"]), {1, 2})
        self.assertEqual(len(comparison["label_mismatches"]), 1)
        self.assertTrue(ground_truths[0].matched)
        self.assertTrue(predictions[0].matched)

    def test_resolve_model_candidate_accepts_training_run_folder(self):
        run_dir = os.path.join(
            os.getcwd(),
            "dumbbell_models",
            "efficientdet_lite0_2026-04-27_19-07-30",
        )
        if not os.path.exists(run_dir):
            self.skipTest("Sample dumbbell model folder is not present.")

        candidate = InferenceTab.resolve_model_candidate(run_dir)
        labels = InferenceTab.read_labels(candidate.labels_path)

        self.assertEqual(candidate.kind, "tflite")
        self.assertTrue(candidate.path.endswith(".tflite"))
        self.assertEqual(labels, ["dumbbell"])


if __name__ == "__main__":
    unittest.main()
