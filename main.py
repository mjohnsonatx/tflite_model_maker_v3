import tempfile
import os

# Redirect temp files to D:
tempfile.tempdir = r"D:\tflite_cache_temp"
os.environ["TMPDIR"] = tempfile.tempdir
os.environ["TEMP"] = tempfile.tempdir
os.environ["TMP"] = tempfile.tempdir

os.makedirs(tempfile.tempdir, exist_ok=True)

import sys
import datetime
import tflite_model_maker
from tflite_model_maker import object_detector

from DualLogger import DualLogger, save_specifications
from PlotTrainingMetrics import visualize_training

# -----------------------------
# Configuration
# -----------------------------
EXPORT_BASE_DIR = 'kettlebell models'
DATA_DIR = 'FINAL_KB_DATA'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

BATCH_SIZE = 32
EPOCHS = 150

BACKBONE = 'efficientnetv2_b3_imagenet'
ARCHITECTURE = 'efficientdet_lite0'
TRAIN_WHOLE_MODEL = True

KETTLEBELL_LABEL_MAP = {1: "kettlebell"}

# All hyperparameters - easier to reference
HPARAMS = {
    # Model architecture
    'backbone_name': BACKBONE,
    'num_classes': 1,
    'image_size': 320,

    # NMS configurations
    'nms_configs': {
        'method': 'gaussian',
        'iou_thresh': 0.5,
        'score_thresh': 0.3,
        'sigma': 0.5,
        'pyfunc': False,
        'max_nms_inputs': 5000,
        'max_output_size': 100,
    },

    # Loss and optimization parameters
    'alpha': 0.25,
    'gamma': 1.5,
    'delta': 0.1,
    'box_loss_weight': 60.0,
    'label_smoothing': 0.1,

    # Learning rate schedule
    'learning_rate': 0.01,
    'lr_warmup_init': 0.001,
    'lr_warmup_epoch': 5.0,
    'lr_decay_method': 'cosine',
    'first_lr_drop_epoch': 20.0,
    'second_lr_drop_epoch': 40.0,
    'poly_lr_power': 0.9,

    # Optimization
    'momentum': 0.9,
    'optimizer': 'sgd',
    'weight_decay': 4e-5,
    'clip_gradients_norm': 10.0,
    'moving_average_decay': 0.9998,

    # Augmentation
    'input_rand_hflip': True,
    'jitter_min': 0.85,
    'jitter_max': 1.15,
    'autoaugment_policy': None,
    'grid_mask': False,

    # Anchors / detection architecture
    'anchor_scale': 3.0,
    'aspect_ratios': [0.5, 1.0, 2.0],
    'num_scales': 3,

    # Training settings
    'skip_crowd_during_training': True,
    'max_instances_per_image': 100,
    'mixed_precision': False,
    'drop_remainder': True,
    'map_freq': 5,
    'num_epochs': 150,
}


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


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Create a dated export directory for this run

    CACHE_DIR = r"D:\tflite_cache"
    # Ensure the directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPORT_DIR = os.path.join(EXPORT_BASE_DIR, f"{ARCHITECTURE}_{date_str}")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Set up logging to file
    log_filepath = os.path.join(EXPORT_DIR, "training_log.txt")

    # Redirect both stdout and stderr to capture all output
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

        # Save specifications
        specs_filepath = save_specifications(
            EXPORT_DIR,
            BATCH_SIZE,
            EPOCHS,
            HPARAMS,
            TRAIN_WHOLE_MODEL,
            BACKBONE,
            ARCHITECTURE
        )
        print(f"Specifications saved to: {specs_filepath}")
        print()

        # -----------------------------
        # Model specification
        # -----------------------------
        model_spec = tflite_model_maker.object_detector.EfficientDetSpec(
            model_name='efficientdet-lite0',
            uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
            hparams=HPARAMS
        )

        # -----------------------------
        # Data loaders
        # -----------------------------
        train = object_detector.DataLoader.from_pascal_voc(
            images_dir=TRAIN_DIR,
            annotations_dir=TRAIN_DIR,
            label_map=KETTLEBELL_LABEL_MAP,
            cache_dir=CACHE_DIR,
            num_shards=4
        )

        valid = object_detector.DataLoader.from_pascal_voc(
            images_dir=VALID_DIR,
            annotations_dir=VALID_DIR,
            label_map=KETTLEBELL_LABEL_MAP,
            cache_dir=CACHE_DIR,
            num_shards=2
        )

        test = object_detector.DataLoader.from_pascal_voc(
            images_dir=TEST_DIR,
            annotations_dir=TEST_DIR,
            label_map=KETTLEBELL_LABEL_MAP,
            cache_dir=CACHE_DIR,
            num_shards=1
        )

        print(f"Training: {train.size} images")
        print(f"Validation: {valid.size} images")
        print(f"Test: {test.size} images")
        print()

        # -----------------------------
        # Create / train model
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

        # Evaluate on validation set
        print("\nValidation Set Evaluation:")
        val_metrics = model.evaluate(valid, batch_size=BATCH_SIZE)
        print(f"Validation metrics: {val_metrics}")

        # Evaluate on test set
        print("\nTest Set Evaluation:")
        test_metrics = model.evaluate(test, batch_size=BATCH_SIZE)
        print(f"Test metrics: {test_metrics}")

        # -----------------------------
        # Export model (TFLite + SavedModel + label map)
        # -----------------------------
        tflite_filename = (
            f"{ARCHITECTURE}_whole.tflite"
            if TRAIN_WHOLE_MODEL
            else f"{ARCHITECTURE}.tflite"
        )

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
        # Restore original stdout/stderr and close log files
        sys.stdout = stdout_logger.terminal
        sys.stderr = stderr_logger.terminal
        stdout_logger.close()
        stderr_logger.close()

        print(f"\n✅ Training log saved to: {log_filepath}")
        print(f"✅ Training specs saved to: {specs_filepath}")
        print("\n" + "=" * 80)
        print("STARTING AUTOMATIC VISUALIZATION...")
        print("=" * 80)

        # Auto-plot the training results
        if visualize_training(log_filepath):
            print("\n" + "=" * 80)
            print("✅ Visualization complete!")
            print("=" * 80)
