import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class EnvConfig:
    """Read local .env values without adding a runtime dependency."""

    TRUE_VALUES = {"1", "true", "yes", "y", "on"}
    FALSE_VALUES = {"0", "false", "no", "n", "off"}
    NONE_VALUES = {"", "none", "null"}

    def __init__(self, env_path: Optional[Path] = None):
        self.env_path = Path(env_path) if env_path else Path(__file__).with_name(".env")
        self.file_values = self._read_env_file(self.env_path)

    @staticmethod
    def _read_env_file(path: Path) -> Dict[str, str]:
        values: Dict[str, str] = {}
        if not path.exists():
            return values

        with path.open("r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                values[key] = value
        return values

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key, self.file_values.get(key, default))

    def require(self, key: str) -> str:
        value = self.get(key)
        if value is None or value.strip() == "":
            raise ValueError(
                f"Missing required environment value: {key}. "
                "Create a local .env from .env.example and fill it in."
            )
        return value

    def text(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        value = self.require(key) if required else self.get(key, default)
        if value is None:
            return None
        return str(value)

    def optional_text(self, key: str, default: Optional[str] = None) -> Optional[str]:
        value = self.get(key, default)
        if value is None or value.strip().lower() in self.NONE_VALUES:
            return None
        return value

    def int(self, key: str, default: Optional[int] = None, required: bool = False) -> int:
        value = self.require(key) if required else self.get(key)
        if value is None:
            if default is None:
                raise ValueError(f"Missing required integer environment value: {key}")
            return default
        return int(value)

    def float(self, key: str, default: Optional[float] = None, required: bool = False) -> float:
        value = self.require(key) if required else self.get(key)
        if value is None:
            if default is None:
                raise ValueError(f"Missing required float environment value: {key}")
            return default
        return float(value)

    def optional_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        value = self.get(key)
        if value is None or value.strip().lower() in self.NONE_VALUES:
            return default
        return float(value)

    def bool(self, key: str, default: Optional[bool] = None, required: bool = False) -> bool:
        value = self.require(key) if required else self.get(key)
        if value is None:
            if default is None:
                raise ValueError(f"Missing required boolean environment value: {key}")
            return default

        normalized = value.strip().lower()
        if normalized in self.TRUE_VALUES:
            return True
        if normalized in self.FALSE_VALUES:
            return False
        raise ValueError(f"Invalid boolean for {key}: {value}")

    def float_list(self, key: str, required: bool = False) -> list[float]:
        value = self.require(key) if required else self.get(key, "")
        cleaned = value.strip().strip("[]")
        if not cleaned:
            return []
        return [float(item.strip()) for item in cleaned.split(",") if item.strip()]


@dataclass
class TrainingSettings:
    temp_dir: str
    cache_dir: str
    data_dir: str
    train_dir: str
    valid_dir: str
    test_dir: str
    export_base_dir: str
    batch_size: int
    epochs: int
    backbone: str
    architecture: str
    train_whole_model: bool
    label: str
    label_map_id: int
    model_name: str
    model_uri: str
    mixed_precision_policy: str
    hparams: Dict[str, Any]
    tflite_max_detections: int


def build_hparams(env: EnvConfig, prefix: str = "TFLITE_HPARAM") -> Dict[str, Any]:
    return {
        "backbone_name": env.text("TFLITE_MODEL_BACKBONE", required=True),
        "num_classes": env.int(f"{prefix}_NUM_CLASSES", required=True),
        "image_size": env.int(f"{prefix}_IMAGE_SIZE", required=True),
        "nms_configs": {
            "method": env.text(f"{prefix}_NMS_METHOD", required=True),
            "iou_thresh": env.float(f"{prefix}_NMS_IOU_THRESH", required=True),
            "score_thresh": env.float(f"{prefix}_NMS_SCORE_THRESH", required=True),
            "sigma": env.optional_float(f"{prefix}_NMS_SIGMA"),
            "pyfunc": env.bool(f"{prefix}_NMS_PYFUNC", required=True),
            "max_nms_inputs": env.int(f"{prefix}_NMS_MAX_INPUTS", required=True),
            "max_output_size": env.int(f"{prefix}_NMS_MAX_OUTPUT_SIZE", required=True),
        },
        "alpha": env.float(f"{prefix}_ALPHA", required=True),
        "gamma": env.float(f"{prefix}_GAMMA", required=True),
        "delta": env.float(f"{prefix}_DELTA", required=True),
        "box_loss_weight": env.float(f"{prefix}_BOX_LOSS_WEIGHT", required=True),
        "label_smoothing": env.float(f"{prefix}_LABEL_SMOOTHING", required=True),
        "learning_rate": env.float(f"{prefix}_LEARNING_RATE", required=True),
        "lr_warmup_init": env.float(f"{prefix}_LR_WARMUP_INIT", required=True),
        "lr_warmup_epoch": env.float(f"{prefix}_LR_WARMUP_EPOCH", required=True),
        "lr_decay_method": env.text(f"{prefix}_LR_DECAY_METHOD", required=True),
        "first_lr_drop_epoch": env.float(f"{prefix}_FIRST_LR_DROP_EPOCH", required=True),
        "second_lr_drop_epoch": env.float(f"{prefix}_SECOND_LR_DROP_EPOCH", required=True),
        "poly_lr_power": env.float(f"{prefix}_POLY_LR_POWER", required=True),
        "momentum": env.float(f"{prefix}_MOMENTUM", required=True),
        "optimizer": env.text(f"{prefix}_OPTIMIZER", required=True),
        "weight_decay": env.float(f"{prefix}_WEIGHT_DECAY", required=True),
        "clip_gradients_norm": env.float(f"{prefix}_CLIP_GRADIENTS_NORM", required=True),
        "moving_average_decay": env.float(f"{prefix}_MOVING_AVERAGE_DECAY", required=True),
        "input_rand_hflip": env.bool(f"{prefix}_INPUT_RAND_HFLIP", required=True),
        "jitter_min": env.float(f"{prefix}_JITTER_MIN", required=True),
        "jitter_max": env.float(f"{prefix}_JITTER_MAX", required=True),
        "autoaugment_policy": env.optional_text(f"{prefix}_AUTOAUGMENT_POLICY"),
        "grid_mask": env.bool(f"{prefix}_GRID_MASK", required=True),
        "anchor_scale": env.float(f"{prefix}_ANCHOR_SCALE", required=True),
        "aspect_ratios": env.float_list(f"{prefix}_ASPECT_RATIOS", required=True),
        "num_scales": env.int(f"{prefix}_NUM_SCALES", required=True),
        "skip_crowd_during_training": env.bool(f"{prefix}_SKIP_CROWD_DURING_TRAINING", required=True),
        "max_instances_per_image": env.int(f"{prefix}_MAX_INSTANCES_PER_IMAGE", required=True),
        "mixed_precision": env.bool(f"{prefix}_MIXED_PRECISION", required=True),
        "drop_remainder": env.bool(f"{prefix}_DROP_REMAINDER", required=True),
        "map_freq": env.int(f"{prefix}_MAP_FREQ", required=True),
        "num_epochs": env.int(f"{prefix}_NUM_EPOCHS", required=True),
    }


def load_training_settings(env_path: Optional[Path] = None) -> TrainingSettings:
    env = EnvConfig(env_path)
    data_dir = env.text("TFLITE_DATA_DIR", required=True)

    return TrainingSettings(
        temp_dir=env.text("TFLITE_TMP_DIR", required=True),
        cache_dir=env.text("TFLITE_CACHE_DIR", required=True),
        data_dir=data_dir,
        train_dir=os.path.join(data_dir, env.text("TFLITE_TRAIN_SUBDIR", required=True)),
        valid_dir=os.path.join(data_dir, env.text("TFLITE_VALID_SUBDIR", required=True)),
        test_dir=os.path.join(data_dir, env.text("TFLITE_TEST_SUBDIR", required=True)),
        export_base_dir=env.text("TFLITE_EXPORT_BASE_DIR", required=True),
        batch_size=env.int("TFLITE_BATCH_SIZE", required=True),
        epochs=env.int("TFLITE_EPOCHS", required=True),
        backbone=env.text("TFLITE_MODEL_BACKBONE", required=True),
        architecture=env.text("TFLITE_MODEL_ARCHITECTURE", required=True),
        train_whole_model=env.bool("TFLITE_TRAIN_WHOLE_MODEL", required=True),
        label=env.text("TFLITE_LABEL", required=True),
        label_map_id=env.int("TFLITE_LABEL_MAP_ID", required=True),
        model_name=env.text("TFLITE_MODEL_NAME", required=True),
        model_uri=env.text("TFLITE_MODEL_URI", required=True),
        mixed_precision_policy=env.text("TFLITE_MIXED_PRECISION_POLICY", required=True),
        hparams=build_hparams(env),
        tflite_max_detections=env.int("TFLITE_TFLITE_MAX_DETECTIONS", required=True)
    )
