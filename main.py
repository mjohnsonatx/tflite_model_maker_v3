import tempfile
import shutil
import sys
import datetime
import os
import random
import numpy as np
from pathlib import Path

from env_config import load_training_settings

SETTINGS = load_training_settings()

"""
Settings to implement determinism in an effort to reduce variability caused by mixed precision
When implemented, this error was thrown:
When determinism is enabled, random ops must have a seed specified.
	 [[{{node blocks_2/random_uniform/RandomUniform}}]] [Op:__inference_train_function_153964]
"""
# os.environ["TF_DETERMINISTIC_OPS"] = "1"   # best-effort determinism in TF 2.x
# os.environ["PYTHONHASHSEED"] = "1337"
#
# random.seed(1337)
# np.random.seed(1337)
# tf.random.set_seed(1337)

# -----------------------------
# Redirect temp files to D:
# -----------------------------
tempfile.tempdir = SETTINGS.temp_dir
os.environ["TMPDIR"] = tempfile.tempdir
os.environ["TEMP"] = tempfile.tempdir
os.environ["TMP"] = tempfile.tempdir
os.makedirs(tempfile.tempdir, exist_ok=True)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy(SETTINGS.mixed_precision_policy)  # TF 2.8 OK

import tensorflow as tf
import tflite_model_maker
from tflite_model_maker import object_detector
from DualLogger import DualLogger, save_specifications
from training_visualizer.core.visualizer import visualize_training

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = SETTINGS.data_dir
EXPORT_BASE_DIR = SETTINGS.export_base_dir

TRAIN_DIR = SETTINGS.train_dir
VALID_DIR = SETTINGS.valid_dir
TEST_DIR = SETTINGS.test_dir

CACHE_DIR = SETTINGS.cache_dir

BATCH_SIZE = SETTINGS.batch_size
EPOCHS = SETTINGS.epochs

BACKBONE = SETTINGS.backbone
ARCHITECTURE = SETTINGS.architecture
TRAIN_WHOLE_MODEL = SETTINGS.train_whole_model

LABEL_MAP = {SETTINGS.label_map_id: SETTINGS.label}
HPARAMS = SETTINGS.hparams
MODEL_NAME = SETTINGS.model_name
MODEL_URI = SETTINGS.model_uri


# -----------------------------
# Utilities
# -----------------------------
def train_model(model, train_data, validation_data, epochs):
    model.train(
        train_data=train_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=BATCH_SIZE
    )
    return model


def clean_cache_dir(path):
    """Remove and recreate a directory safely."""
    if os.path.exists(path):
        print(f"🧹 Clearing old cache: {path}")
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"⚠️  Could not fully remove cache ({e}), continuing...")
    os.makedirs(path, exist_ok=True)
    print(f"✅ Cache directory ready: {path}")


def check_gpu_status():
    import os, tensorflow as tf
    print("=" * 80)
    print("🔍 GPU VERIFICATION")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"TF_GPU_ALLOCATOR: {os.environ.get('TF_GPU_ALLOCATOR', 'default')}\n")

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU detected. Training will run on CPU.")
    else:
        print(f"✅ Detected {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  ↳ memory_growth: enabled")
            except Exception as e:
                print(f"  ↳ memory_growth: failed ({e})")

        try:
            from tensorflow.python.client import device_lib
            for d in device_lib.list_local_devices():
                if d.device_type == 'GPU':
                    print(f"  ↳ {d.physical_device_desc}")
        except Exception:
            pass

    try:
        print("\nVerifying cuDNN runtime version...")
        from tensorflow.python.platform import build_info
        print("  CUDA:", build_info.build_info.get('cuda_version', 'Unknown'))
        print("  cuDNN:", build_info.build_info.get('cudnn_version', 'Unknown'))
    except Exception:
        print("⚠️  Could not query build info (TensorFlow internal).")

    # 🧩 Mixed Precision Status
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.global_policy()
        print(f"\nMixed Precision Policy: {policy.name}")
    except Exception as e:
        print(f"\nMixed Precision Policy: unavailable ({e})")

    print("=" * 80)
    print()


def visualize_training_results(log_filepath, spec_filepath=None):
    """Launch the training visualizer for the completed training."""
    import subprocess
    import sys

    # Find the run_visualizer.py script
    # Adjust this path based on where run_visualizer.py is relative to your training script
    visualizer_path = Path(__file__).parent / "run_visualizer.py"

    if not visualizer_path.exists():
        # Try parent directory
        visualizer_path = Path(__file__).parent.parent / "run_visualizer.py"

    if visualizer_path.exists():
        # Run the visualizer as a subprocess
        cmd = [sys.executable, str(visualizer_path), log_filepath]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"Visualizer error: {result.stderr}")
            return False
    else:
        print(f"Could not find run_visualizer.py")
        return False


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    # Clean temp and cache dirs
    clean_cache_dir(tempfile.tempdir)
    clean_cache_dir(CACHE_DIR)

    # Check GPU setup before any heavy work
    check_gpu_status()

    # Create dated export directory
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPORT_DIR = os.path.join(EXPORT_BASE_DIR, f"{ARCHITECTURE}_{date_str}")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Set up logging
    log_filepath = os.path.join(EXPORT_DIR, "training_log.txt")
    stdout_logger = DualLogger(log_filepath, 'w')
    stderr_logger = DualLogger(log_filepath, 'a')
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    try:
        print("=" * 80)
        print(f"Training Session Started: {date_str}")
        print("=" * 80)
        print(f"Export directory: {EXPORT_DIR}")
        print(f"Log file: {log_filepath}")
        print("=" * 80)
        print()

        specs_filepath = save_specifications(
            export_dir=EXPORT_DIR,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            hparams=HPARAMS,
            train_whole_model=TRAIN_WHOLE_MODEL,
            backbone=BACKBONE,
            architecture=ARCHITECTURE,
            mixed_precision_policy=mixed_precision.global_policy().name
        )
        print(f"Specifications saved to: {specs_filepath}\n")

        # -----------------------------
        # Model spec
        # -----------------------------
        model_spec = tflite_model_maker.object_detector.EfficientDetSpec(
            model_name=MODEL_NAME,
            uri=MODEL_URI,
            hparams=HPARAMS
        )

        # -----------------------------
        # Data loaders
        # -----------------------------
        train = object_detector.DataLoader.from_pascal_voc(
            images_dir=TRAIN_DIR,
            annotations_dir=TRAIN_DIR,
            label_map=LABEL_MAP,
            cache_dir=CACHE_DIR,
            num_shards=4
        )
        valid = object_detector.DataLoader.from_pascal_voc(
            images_dir=VALID_DIR,
            annotations_dir=VALID_DIR,
            label_map=LABEL_MAP,
            cache_dir=CACHE_DIR,
            num_shards=2
        )
        test = object_detector.DataLoader.from_pascal_voc(
            images_dir=TEST_DIR,
            annotations_dir=TEST_DIR,
            label_map=LABEL_MAP,
            cache_dir=CACHE_DIR,
            num_shards=1
        )

        print(f"Training: {train.size} images")
        print(f"Validation: {valid.size} images")
        print(f"Test: {test.size} images\n")

        # -----------------------------
        # Create / train_NEEDS_ANOTHER_PASS model
        # -----------------------------
        print("Initializing model...")
        model = object_detector.create(
            train_data=train,
            model_spec=model_spec,
            batch_size=BATCH_SIZE,
            train_whole_model=TRAIN_WHOLE_MODEL,
            validation_data=valid,
            epochs=0,
            do_train=False
        )

        print("Training model...")
        model = train_model(model, train, valid, EPOCHS)

        # -----------------------------
        # Evaluate model
        # -----------------------------
        print("\nEvaluating model...")
        print("\nValidation Set Evaluation:")
        val_metrics = model.evaluate(valid, batch_size=BATCH_SIZE)
        print(f"Validation metrics: {val_metrics}")

        print("\nTest Set Evaluation:")
        test_metrics = model.evaluate(test, batch_size=BATCH_SIZE)
        print(f"Test metrics: {test_metrics}")

        # -----------------------------
        # Export model
        # -----------------------------
        label_token = SETTINGS.label.upper().replace(" ", "_")
        tflite_filename = f"__{label_token}__{ARCHITECTURE}_{date_str}.tflite"

        print(f"\nExporting model to {EXPORT_DIR}...")
        model.export(
            export_dir=EXPORT_DIR,
            tflite_filename=tflite_filename,
            export_format=[
                tflite_model_maker.ExportFormat.TFLITE,
                tflite_model_maker.ExportFormat.SAVED_MODEL,
                tflite_model_maker.ExportFormat.LABEL
            ]
        )

        print("\n" + "=" * 80)
        print(f"✅ Export complete! Assets saved to:\n{EXPORT_DIR}")
        print("=" * 80)
        print("This folder now contains:")
        print("  • The .tflite model")
        print("  • A full SavedModel (for reloading)")
        print("  • The label map")
        print(f"  • Training log: {log_filepath}")
        print(f"  • Training specs: {specs_filepath}")
        print("=" * 80)

        end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"\nTraining Session Ended: {end_time}")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR OCCURRED")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        print("=" * 80)
        raise

    finally:
        sys.stdout = stdout_logger.terminal
        sys.stderr = stderr_logger.terminal
        stdout_logger.close()
        stderr_logger.close()

        print(f"\n✅ Training log saved to: {log_filepath}")
        print(f"✅ Training specs saved to: {specs_filepath}")
        print("\n" + "=" * 80)
        print("STARTING AUTOMATIC VISUALIZATION...")
        print("=" * 80)

        if visualize_training_results(log_filepath, specs_filepath):
            print("\n" + "=" * 80)
            print("✅ Visualization complete!")
            print("=" * 80)
