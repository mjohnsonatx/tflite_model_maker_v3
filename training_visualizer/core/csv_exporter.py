"""CSV export utilities for training visualizer data."""
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..analysis.statistics import calculate_statistics
from ..parsers.log_parser import parse_training_log
from ..parsers.spec_parser import parse_model_specs


METRIC_COLUMNS = [
    "det_loss",
    "cls_loss",
    "box_loss",
    "reg_l2_loss",
    "loss",
    "learning_rate",
    "gradient_norm",
    "val_det_loss",
    "val_cls_loss",
    "val_box_loss",
    "val_reg_l2_loss",
    "val_loss",
]

DERIVED_COLUMNS = [
    "train_val_loss_gap",
    "loss_change_pct",
    "val_loss_change_pct",
]

STATS_COLUMNS = [
    "total_epochs",
    "final_train_loss",
    "final_val_loss",
    "best_val_loss",
    "best_epoch",
    "train_loss_reduction",
    "val_loss_reduction",
    "final_overfit_gap",
    "avg_overfit_gap",
    "train_stability",
    "val_stability",
    "initial_lr",
    "final_lr",
    "lr_reduction",
    "val_ap",
    "val_ap50",
    "val_ap75",
    "test_ap",
    "test_ap50",
    "test_ap75",
]

EVAL_METRIC_MAPPING = {
    "DetectionBoxes_Precision/mAP": "AP",
    "DetectionBoxes_Precision/mAP@.50IOU": "AP50",
    "DetectionBoxes_Precision/mAP@.75IOU": "AP75",
    "DetectionBoxes_Precision/mAP (small)": "APs",
    "DetectionBoxes_Precision/mAP (medium)": "APm",
    "DetectionBoxes_Precision/mAP (large)": "APl",
    "DetectionBoxes_Recall/AR@1": "ARmax1",
    "DetectionBoxes_Recall/AR@10": "ARmax10",
    "DetectionBoxes_Recall/AR@100": "ARmax100",
    "DetectionBoxes_Recall/AR@100 (small)": "ARs",
    "DetectionBoxes_Recall/AR@100 (medium)": "ARm",
    "DetectionBoxes_Recall/AR@100 (large)": "ARl",
}


def export_training_data_csv(
    model_configs: List[Dict[str, str]],
    output_path: Optional[str] = None,
) -> Tuple[Path, int]:
    """
    Export parsed visualizer data to one LLM-friendly CSV.

    Each row represents one epoch for one dataset. Run-level data such as
    statistics, evaluation metrics, model specs, and hyperparameters is repeated
    on each row so the CSV remains a single self-contained table.
    """
    if not model_configs:
        raise ValueError("No training logs were provided for CSV export.")

    datasets = [_load_dataset(config) for config in model_configs]
    rows, columns = _build_rows(datasets)

    csv_path = _resolve_output_path(output_path, datasets)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})

    return csv_path, len(rows)


def _load_dataset(config: Dict[str, str]) -> Dict[str, Any]:
    log_path = Path(config["path"]).expanduser()
    name = config.get("name") or log_path.parent.name or log_path.stem
    spec_path = Path(config["spec_path"]).expanduser() if config.get("spec_path") else log_path.parent / "training_specs.txt"

    metrics, eval_metrics = parse_training_log(log_path)
    specs = parse_model_specs(spec_path) if spec_path.exists() else {"main": {}, "hyperparameters": {}}
    stats = calculate_statistics(metrics, eval_metrics)

    return {
        "name": name,
        "log_path": log_path,
        "spec_path": spec_path if spec_path.exists() else None,
        "metrics": metrics,
        "eval_metrics": eval_metrics,
        "specs": specs,
        "stats": stats,
    }


def _build_rows(datasets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    columns: List[str] = ["dataset", "log_file", "spec_file", "row_index", "epoch"]

    for metric in METRIC_COLUMNS + DERIVED_COLUMNS:
        _append_column(columns, metric)

    for stat in STATS_COLUMNS:
        _append_column(columns, f"stat_{stat}")

    for dataset in datasets:
        for row in _dataset_rows(dataset):
            rows.append(row)
            for column in row:
                _append_column(columns, column)

    return rows, columns


def _dataset_rows(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics = dataset["metrics"]
    stats = dataset["stats"]
    eval_metrics = dataset["eval_metrics"]
    specs = dataset["specs"]
    epoch_count = max([len(values) for values in metrics.values() if isinstance(values, list)] or [0])
    row_count = max(1, epoch_count)
    rows = []

    for index in range(row_count):
        row = {
            "dataset": dataset["name"],
            "log_file": str(dataset["log_path"]),
            "spec_file": str(dataset["spec_path"]) if dataset["spec_path"] else "",
            "row_index": index + 1,
            "epoch": _metric_value(metrics, "epoch", index),
        }

        for metric in METRIC_COLUMNS:
            row[metric] = _metric_value(metrics, metric, index)

        row["train_val_loss_gap"] = _difference(row.get("loss"), row.get("val_loss"))
        row["loss_change_pct"] = _change_pct(metrics, "loss", index)
        row["val_loss_change_pct"] = _change_pct(metrics, "val_loss", index)

        for stat_key, stat_value in stats.items():
            row[f"stat_{_column_key(stat_key)}"] = stat_value

        for split, split_metrics in eval_metrics.items():
            split_key = "test" if split == "test_CLEAN" else _column_key(split)
            for metric_key, metric_value in _normalized_eval_metrics(split_metrics).items():
                row[f"eval_{split_key}_{_column_key(metric_key)}"] = metric_value

        for spec_key, spec_value in specs.get("main", {}).items():
            row[f"spec_{_column_key(spec_key)}"] = spec_value

        for spec_key, spec_value in specs.get("hyperparameters", {}).items():
            row[f"hyperparameter_{_column_key(spec_key)}"] = spec_value

        rows.append(row)

    return rows


def _normalized_eval_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for metric_key, metric_value in metrics.items():
        normalized_key = EVAL_METRIC_MAPPING.get(metric_key, metric_key)
        normalized[normalized_key] = metric_value
    return normalized


def _resolve_output_path(output_path: Optional[str], datasets: List[Dict[str, Any]]) -> Path:
    if output_path:
        path = Path(output_path).expanduser()
        if path.exists() and path.is_dir():
            return path / _default_filename(datasets)
        if not path.suffix:
            path = path.with_suffix(".csv")
        return path

    if len(datasets) == 1:
        log_path = datasets[0]["log_path"]
        return log_path.with_name(f"{log_path.stem}_visualizer_data.csv")

    return Path.cwd() / "training_visualizer_data.csv"


def _default_filename(datasets: List[Dict[str, Any]]) -> str:
    if len(datasets) == 1:
        return f"{datasets[0]['log_path'].stem}_visualizer_data.csv"
    return "training_visualizer_data.csv"


def _metric_value(metrics: Dict[str, List[Any]], metric: str, index: int) -> Any:
    values = metrics.get(metric, [])
    if index < len(values):
        return values[index]
    return ""


def _difference(left: Any, right: Any) -> Any:
    if left == "" or right == "":
        return ""
    return left - right


def _change_pct(metrics: Dict[str, List[Any]], metric: str, index: int) -> Any:
    values = metrics.get(metric, [])
    if index <= 0 or index >= len(values):
        return ""

    previous = values[index - 1]
    current = values[index]
    if previous == 0:
        return ""
    return (current - previous) / previous * 100


def _column_key(value: Any) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return cleaned or "value"


def _append_column(columns: List[str], column: str) -> None:
    if column not in columns:
        columns.append(column)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value
