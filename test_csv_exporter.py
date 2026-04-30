import csv
import os
import unittest
from pathlib import Path

from training_visualizer.core.csv_exporter import export_training_data_csv


class CsvExporterTests(unittest.TestCase):
    def test_export_training_data_csv_flattens_visualizer_data(self):
        temp_root = Path(os.getcwd()) / "codex_csv_exporter_test"
        run_dir = temp_root / "csv_exporter_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "training_log.txt"
        spec_path = run_dir / "training_specs.txt"
        output_path = run_dir / "visualizer_data.csv"
        for path in (log_path, spec_path, output_path):
            path.unlink(missing_ok=True)
        self.addCleanup(_cleanup_files, log_path, spec_path, output_path, run_dir, temp_root)

        log_path.write_text(
            "\n".join(
                [
                    "Epoch 1/2",
                    "1/1 [==============================] - 1s 1s/step - det_loss: 1.0000 - cls_loss: 0.5000 - box_loss: 0.0100 - reg_l2_loss: 0.1000 - loss: 1.1000 - learning_rate: 1.0000e-03 - gradient_norm: 2.0000 - val_det_loss: 1.2000 - val_cls_loss: 0.6000 - val_box_loss: 0.0200 - val_reg_l2_loss: 0.1000 - val_loss: 1.3000",
                    "Epoch 2/2",
                    "1/1 [==============================] - 1s 1s/step - det_loss: 0.8000 - cls_loss: 0.4000 - box_loss: 0.0080 - reg_l2_loss: 0.1000 - loss: 0.9000 - learning_rate: 5.0000e-04 - gradient_norm: 1.5000 - val_det_loss: 1.0000 - val_cls_loss: 0.5000 - val_box_loss: 0.0150 - val_reg_l2_loss: 0.1000 - val_loss: 1.1000",
                    "Validation metrics: {'AP': 0.25, 'AP50': 0.50}",
                    "Test metrics: {'AP': 0.20, 'DetectionBoxes_Precision/mAP@.50IOU': 0.45}",
                ]
            ),
            encoding="utf-8",
        )
        spec_path.write_text(
            "\n".join(
                [
                    "Batch Size: 4",
                    "Epochs: 2",
                    "Architecture: efficientdet_lite0",
                    "Model Name: demo-model",
                    "Train Whole Model: True",
                    "learning_rate: 0.001",
                    "optimizer: sgd",
                ]
            ),
            encoding="utf-8",
        )

        csv_path, row_count = export_training_data_csv(
            [{"path": str(log_path), "name": "run_a"}],
            str(output_path),
        )

        self.assertEqual(csv_path, output_path)
        self.assertEqual(row_count, 2)

        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dataset"], "run_a")
        self.assertEqual(rows[0]["epoch"], "1")
        self.assertEqual(rows[0]["loss"], "1.1")
        self.assertEqual(rows[0]["val_loss"], "1.3")
        self.assertEqual(rows[0]["train_val_loss_gap"], "-0.19999999999999996")
        self.assertEqual(rows[1]["loss_change_pct"], "-18.181818181818183")
        self.assertEqual(rows[0]["stat_best_epoch"], "2")
        self.assertEqual(rows[0]["eval_validation_ap50"], "0.5")
        self.assertEqual(rows[0]["eval_test_ap50"], "0.45")
        self.assertEqual(rows[0]["spec_batch_size"], "4")
        self.assertEqual(rows[0]["hyperparameter_learning_rate"], "0.001")


def _cleanup_files(*paths):
    for path in paths[:-2]:
        path.unlink(missing_ok=True)
    for path in paths[-2:]:
        try:
            path.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
