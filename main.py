import os
import tensorflow as tf

from tflite_model_maker import object_detector
from tflite_model_maker import model_spec

# Disable training on GPU.
# tf.config.set_visible_devices([], 'GPU')

EXPORT_DIR = 'models'
DATA_DIR = 'new data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
BATCH_SIZE = 4
ARCHITECTURE = 'efficientdet_lite0'
TRAIN_WHOLE_MODEL = True

if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)

    spec = model_spec.get(ARCHITECTURE)

    train = object_detector.DataLoader.from_pascal_voc(
        images_dir=TRAIN_DIR,
        annotations_dir=TRAIN_DIR,
        label_map={1: "Barbell"}
    )

    valid = object_detector.DataLoader.from_pascal_voc(
        images_dir=VALID_DIR,
        annotations_dir=VALID_DIR,
        label_map={1: "Barbell"}
    )

    test = object_detector.DataLoader.from_pascal_voc(
        images_dir=TEST_DIR,
        annotations_dir=TEST_DIR,
        label_map={1: "Barbell"}
    )

    model = object_detector.create(
        train,
        epochs=50,
        model_spec=spec,
        batch_size=BATCH_SIZE,
        train_whole_model=TRAIN_WHOLE_MODEL,
        validation_data=valid
    )

    tflite_filename = f'{ARCHITECTURE}.tflite'

    if TRAIN_WHOLE_MODEL:
        tflite_filename = f'{ARCHITECTURE}_whole.tflite'

    print("Evaluating the original model...")
    print(model.evaluate(test, batch_size=BATCH_SIZE))

    print("Exporting the model...")
    model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)

    print("Evaluating the exported model...")
    print(model.evaluate_tflite(os.path.join(EXPORT_DIR, tflite_filename), test))