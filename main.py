# import os
# import tensorflow as tf
#
# from tflite_model_maker import object_detector
# from tflite_model_maker import model_spec
#
# # Disable training on GPU.
# # tf.config.set_visible_devices([], 'GPU')
#
# EXPORT_DIR = 'models'
# DATA_DIR = 'old data'
# TRAIN_DIR = os.path.join(DATA_DIR, 'train')
# VALID_DIR = os.path.join(DATA_DIR, 'valid')
# TEST_DIR = os.path.join(DATA_DIR, 'test')
# BATCH_SIZE = 16
# ARCHITECTURE = 'efficientdet_lite0'
# TRAIN_WHOLE_MODEL = True
#
# if __name__ == "__main__":
#     os.makedirs(EXPORT_DIR, exist_ok=True)
#
#     spec = model_spec.get(ARCHITECTURE)
#
#     train = object_detector.DataLoader.from_pascal_voc(
#         images_dir=TRAIN_DIR,
#         annotations_dir=TRAIN_DIR,
#         label_map={1: "Barbell"}
#     )
#
#     valid = object_detector.DataLoader.from_pascal_voc(
#         images_dir=VALID_DIR,
#         annotations_dir=VALID_DIR,
#         label_map={1: "Barbell"}
#     )
#
#     test = object_detector.DataLoader.from_pascal_voc(
#         images_dir=TEST_DIR,
#         annotations_dir=TEST_DIR,
#         label_map={1: "Barbell"}
#     )
#
#     model = object_detector.create(
#         train,
#         epochs=75,
#         model_spec=spec,
#         batch_size=BATCH_SIZE,
#         train_whole_model=TRAIN_WHOLE_MODEL,
#         validation_data=valid
#     )
#
#     tflite_filename = f'{ARCHITECTURE}.tflite'
#
#     if TRAIN_WHOLE_MODEL:
#         tflite_filename = f'{ARCHITECTURE}_whole.tflite'
#
#     print("Exporting the model...")
#     model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)
#
#     print("Quantizing the model...")
#     quantized_tflite_filename = f'{ARCHITECTURE}_quantized_4_75.tflite'
#     converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     quantized_tflite_model = converter.convert()
#
#     with open(os.path.join(EXPORT_DIR, quantized_tflite_filename), 'wb') as f:
#         f.write(quantized_tflite_model)
#
#     with open(os.path.join(EXPORT_DIR, quantized_tflite_filename), 'wb') as f:
#         f.write(quantized_tflite_model)
import csv
# print("Evaluating the quantized model...")
# print(model.evaluate_tflite(os.path.join(EXPORT_DIR, quantized_tflite_filename), test))

import os
import tensorflow as tf

from tflite_model_maker import object_detector
from tflite_model_maker import model_spec
import numpy as np


EXPORT_DIR = 'models'
DATA_DIR = 'data'
TRAIN_DIR = 'combined data with original data'
#TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
BATCH_SIZE = 32
ARCHITECTURE = 'efficientdet_lite0'
TRAIN_WHOLE_MODEL = True

def print_model_state_summary(model):
    total = 0
    for variable in model.model.trainable_variables:
        total += np.sum(variable.numpy())
    print(f"Sum of all trainable variables: {total}")


if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)

    spec = model_spec.get(ARCHITECTURE)

    train = object_detector.DataLoader.from_pascal_voc(
        images_dir=TRAIN_DIR,
        annotations_dir=TRAIN_DIR,
        label_map={1: "barbell"}
    )

    valid = object_detector.DataLoader.from_pascal_voc(
        images_dir=VALID_DIR,
        annotations_dir=VALID_DIR,
        label_map={1: "barbell"}
    )

    test = object_detector.DataLoader.from_pascal_voc(
        images_dir=TEST_DIR,
        annotations_dir=TEST_DIR,
        label_map={1: "barbell"}
    )

    print(f"Number of training images: {train.size}")
    print(f"Number of validation images: {valid.size}")
    print(f"Number of test images: {test.size}")

    model = object_detector.create(
        train_data=train,
        model_spec=spec,
        validation_data=valid,
        batch_size=BATCH_SIZE,
        train_whole_model=TRAIN_WHOLE_MODEL,
        do_train=False
    )

    tflite_filename = f'{ARCHITECTURE}.tflite'

    if TRAIN_WHOLE_MODEL:
        tflite_filename = f'{ARCHITECTURE}_whole.tflite'

    # Define early stopping parameters
    patience = 10
    min_delta = 0.001

    # Custom training loop
    best_ap = 0  # Initialize with 0 as we want to maximize AP
    epochs = 100  # Set this to your desired number of epochs
    patience_counter = 0
    best_epoch = 0
    global_epoch = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train for one epoch
        model.train(train, validation_data=valid, epochs=1)

        # Evaluate the model to get the validation metrics
        eval_results = model.evaluate(valid)
        val_ap = eval_results['AP']

        print(f"Validation AP: {val_ap}")

        # Check if this is the best model so far
        if val_ap > best_ap + min_delta:
            best_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Check early stopping condition, but only after 50 epochs
        if epoch >= 50 and patience_counter >= patience:
            print(f"Early stopping triggered. Best epoch was {best_epoch + 1} with AP: {best_ap}")
            # Export the model only when early stopping is triggered
            model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)
            break

        # If we didn't trigger early stopping, export the final model
        if epoch == epochs - 1:
            print(f"Reached maximum epochs. Best epoch was {best_epoch + 1} with AP: {best_ap}")
            model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)

    print("Evaluating the model...")
    print(model.evaluate(test, batch_size=BATCH_SIZE))

    #
    # print("Evaluating the original model...")
    # print(model.evaluate(test, batch_size=BATCH_SIZE))
    #
    # print("Exporting the model...")
    # model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)

""""
Example of 8bit quant
"""
#config = QuantizationConfig.for_int8()

# Export your model with UINT8 quantization
#model.export(export_dir='.', tflite_filename='model_quantized_uint8.tflite', quantization_config=config)