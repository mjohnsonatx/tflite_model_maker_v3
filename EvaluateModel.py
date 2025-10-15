import os
import glob
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import cv2
import tflite_support
from tensorflow_examples.lite.model_maker.core.task.object_detector import ObjectDetector

# -----------------------------
# Config
# -----------------------------
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import EfficientDetSpec

# --- Paths ---
EXPORT_DIR = "kettlebell models"
SAVED_MODEL_DIR = f"{EXPORT_DIR}/saved_model"
TEST_DIR = "FINAL_KB_DATA/test"
KETTLEBELL_LABEL_MAP = {1: "kettlebell"}
BATCH_SIZE = 4

# --- Rebuild test dataset ---
test_data = object_detector.DataLoader.from_pascal_voc(
    images_dir=TEST_DIR,
    annotations_dir=TEST_DIR,
    label_map=KETTLEBELL_LABEL_MAP
)

# --- Load model from SavedModel directory ---
spec = EfficientDetSpec(
    model_name='efficientdet-lite0',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
    hparams={'num_classes': 1, 'image_size': 320},
    model_dir=SAVED_MODEL_DIR  # key part!
)

model = object_detector.create(
    train_data=test_data,  # dummy dataset just to init; will not be trained
    model_spec=spec,
    batch_size=BATCH_SIZE,
    train_whole_model=False,
    validation_data=None,
    epochs=0,
    do_train=False
)

# --- Evaluate exactly like before ---
metrics = model.evaluate(test_data, batch_size=BATCH_SIZE)
print("Evaluation metrics:", metrics)


# import tensorflow as tf
# import numpy as np
#
# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path='kettlebell models/efficientdet_lite0_whole.tflite')
# interpreter.allocate_tensors()
#
# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
#
# def run_inference(image):
#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details[0]['index'])
#     return output
#
#
# # Evaluate the model using the test data
# def evaluate_tflite_model(test_data, interpreter):
#     num_correct = 0
#     num_total = 0
#
#     for image, label in test_data:
#         # Preprocess the image and convert to the appropriate input shape
#         image = np.expand_dims(image, axis=0).astype(np.float32)
#
#         # Run inference
#         output = run_inference(image)
#
#         # Postprocess the output to get the predicted label
#         # Assuming this is a classification task; adjust accordingly for detection.
#         predicted_label = np.argmax(output)
#
#         # Compare the predicted label with the true label
#         if predicted_label == label:
#             num_correct += 1
#         num_total += 1
#
#     accuracy = num_correct / num_total
#     return accuracy
#
#
# # Assuming 'test' is your test dataset
# accuracy = evaluate_tflite_model('FINAL_KB_DATA/test', interpreter)
# print(f"Re-evaluated accuracy of the TFLite model: {accuracy:.4f}")