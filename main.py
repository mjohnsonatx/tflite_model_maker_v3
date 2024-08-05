import os
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.task.configs import QuantizationConfig

from tflite_model_maker import object_detector
from tflite_model_maker import model_spec
import numpy as np


EXPORT_DIR = 'models'
DATA_DIR = 'data'
TRAIN_DIR = 'combined data with original data'
#TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
BATCH_SIZE = 2
ARCHITECTURE = 'efficientdet_lite0'
TRAIN_WHOLE_MODEL = True

if __name__ == "__main__":
    # os.makedirs(EXPORT_DIR, exist_ok=True)
    #
    # spec = model_spec.get(ARCHITECTURE)
    #
    # train = object_detector.DataLoader.from_pascal_voc(
    #     images_dir=TRAIN_DIR,
    #     annotations_dir=TRAIN_DIR,
    #     label_map={1: "barbell"}
    # )
    #
    # valid = object_detector.DataLoader.from_pascal_voc(
    #     images_dir=VALID_DIR,
    #     annotations_dir=VALID_DIR,
    #     label_map={1: "barbell"}
    # )
    #
    # test = object_detector.DataLoader.from_pascal_voc(
    #     images_dir=TEST_DIR,
    #     annotations_dir=TEST_DIR,
    #     label_map={1: "barbell"}
    # )
    #
    # print(f"Number of training images: {train.size}")
    # print(f"Number of validation images: {valid.size}")
    # print(f"Number of test images: {test.size}")
    #
    # model = object_detector.create(
    #     train_data=train,
    #     epochs = 65,
    #     model_spec=spec,
    #     validation_data=valid,
    #     batch_size=BATCH_SIZE,
    #     train_whole_model=TRAIN_WHOLE_MODEL,
    #     do_train=True
    # )
    #
    # tflite_filename = f'{ARCHITECTURE}.tflite'
    #
    # if TRAIN_WHOLE_MODEL:
    #     tflite_filename = f'{ARCHITECTURE}_whole.tflite'
    #
    #
    # print("Evaluating the original model...")
    # print(model.evaluate(test, batch_size=BATCH_SIZE))
    #
    # print("Exporting the model...")
    # model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)
    #
    # print("Exporting a quantized model...")

    

    # need to add representative data
    quantization_config = QuantizationConfig(
        inference_input_type=tf.float32,  # Set input tensor to UINT8
        inference_output_type=tf.float32  # Set output tensor to UINT8
    )
    model.export(export_dir=EXPORT_DIR, tflite_filename='model_quantized_uint8.tflite', quantization_config=quantization_config)

""""
Example for representive data
"""

# def load_and_preprocess_image(path):
#     image = tf.io.read_file(path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [224, 224])  # Resize to the input size of the model
#     image = image / 255.0  # Normalize to [0, 1]
#     return image
#
# def representative_data_gen():
#     for image_path in representative_image_paths:
#         # Load and preprocess the image
#         processed_image = load_and_preprocess_image(image_path)
#         # Batch size is 1
#         yield [processed_image.numpy()]
#
# # Assuming 'representative_image_paths' is a list of file paths to your images
# representative_dataset = tf.data.Dataset.from_tensor_slices(representative_image_paths)
# representative_dataset = representative_dataset.map(load_and_preprocess_image).batch(1)

"""
attempted early stopping
"""

# def print_model_state_summary(model):
#     total = 0
#     for variable in model.model.trainable_variables:
#         total += np.sum(variable.numpy())
#     print(f"Sum of all trainable variables: {total}")

#patience = 10
# min_delta = 0.001
#
# # Custom training loop
# best_ap = 0  # Initialize with 0 as we want to maximize AP
# epochs = 100  # Set this to your desired number of epochs
# patience_counter = 0
# best_epoch = 0
# global_epoch = 0
#
# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1}/{epochs}")
#
#     # Train for one epoch
#     model.train(train, validation_data=valid, epochs=1)
#
#     # Evaluate the model to get the validation metrics
#     eval_results = model.evaluate(valid)
#     val_ap = eval_results['AP']
#
#     print(f"Validation AP: {val_ap}")
#
#     # Check if this is the best model so far
#     if val_ap > best_ap + min_delta:
#         best_ap = val_ap
#         best_epoch = epoch
#         patience_counter = 0
#     else:
#         patience_counter += 1
#
#     # Check early stopping condition, but only after 50 epochs
#     if epoch >= 50 and patience_counter >= patience:
#         print(f"Early stopping triggered. Best epoch was {best_epoch + 1} with AP: {best_ap}")
#         # Export the model only when early stopping is triggered
#         model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)
#         break
#
#     # If we didn't trigger early stopping, export the final model
#     if epoch == epochs - 1:
#         print(f"Reached maximum epochs. Best epoch was {best_epoch + 1} with AP: {best_ap}")
#         model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)
#
# print("Evaluating the model...")
# print(model.evaluate(test, batch_size=BATCH_SIZE))