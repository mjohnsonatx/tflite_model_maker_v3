import logging
import os

import numpy as np
import tensorflow as tf
import tflite_model_maker
from PIL import Image
from tflite_model_maker import object_detector, model_spec
import tensorflow_model_optimization as tfmot

EXPORT_DIR = 'kettlebell models'
DATA_DIR = 'Kettlebell Data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train with augment')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test with augment')

BATCH_SIZE = 32
EPOCHS = 400
BACKBONE = 'efficientnetv2_b3_imagenet'
#BACKBONE = 'efficientnet-b3'
ARCHITECTURE = 'efficientdet_lite0'
TRAIN_WHOLE_MODEL = True

LABEL_MAP = {1: "barbell"}
KETTLEBELL_LABEL_MAP = {1: "kettlebell"}


def train_model(model, train_data, validation_data, epochs):
    model.train(
        train_data=train_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=BATCH_SIZE
    )
    return model


if __name__ == "__main__":

    # if os.path.exists(SAVED_MODEL_PATH):
    #     print("Loading existing model...")
    #     loaded_model = tf.saved_model.load('models/v1/saved_model')
    #
    # Create the model spec

    # model_url = 'https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1'
    #
    # # Load the model (this will download it automatically if not cached)
    # model = hub.load(model_url)
    # print("Model downloaded and loaded successfully!")

    # model_spec = tflite_model_maker.object_detector.EfficientDetSpec(
    #     model_name='efficientdet-lite0',
    #     uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
    #     hparams={'backbone_name': BACKBONE}
    #)

    model_spec = tflite_model_maker.object_detector.EfficientDetSpec(
        model_name='efficientdet-lite0',
        uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
        hparams={
            'backbone_name': BACKBONE,
            'nms_configs': {
                'method': 'gaussian',
                'iou_thresh': 0.5,
                'score_thresh': 0.7,
                'sigma': 0.3,
                'pyfunc': False,
                'max_nms_inputs': 5000,
                'max_output_size': 100
            },
            'gamma': 1.25,
            'label_smoothing': 0.1,
            'weight_decay': 4e-5,
            'learning_rate': 0.012,
            'lr_warmup_init': 0.008,
            'first_lr_drop_epoch': 70.0,  # Adjusted to an earlier epoch
            'second_lr_drop_epoch': 90.0,  # Adjusted to follow the first drop sooner
            'num_epochs': 100,
            'momentum': 0.9,
            'optimizer': 'sgd',
            'input_rand_hflip': True,
            'jitter_min': 0.6,
            'jitter_max': 1.4,
            'autoaugment_policy': 'v2',
            'clip_gradients_norm': 10.0,  # Here's the gradient clipping addition
            'anchor_scale': 4.0,
            'aspect_ratios': [1.0, 2.0, 0.5]
        }
    )

    # model_spec = tflite_model_maker.object_detector.EfficientDetSpec(
    #     model_name='efficientdet-lite0',
    #     uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
    #     hparams={
    #         'backbone_name': 'efficientnetv2_b3_imagenet',
    #         'nms_configs': {
    #             'method': 'gaussian',
    #             'iou_thresh': None,
    #             'score_thresh': 0.6,
    #             'sigma': None,
    #             'pyfunc': False,
    #             'max_nms_inputs': 5000,
    #             'max_output_size': 100
    #         },
    #         'gamma': 1.25,
    #         'label_smoothing': 0.1,
    #         'weight_decay': 4e-5,
    #         'learning_rate': 0.012,  # Initial learning rate adjusted
    #         'lr_warmup_init': 0.008,
    #         'first_lr_drop_epoch': 60,  # Adjusted for longer training
    #         'second_lr_drop_epoch': 80,  # Adjusted for longer training
    #         'num_epochs': 100,
    #         'momentum': 0.9,
    #         'optimizer': 'sgd',
    #         'input_rand_hflip': True,
    #         'jitter_min': 0.6,
    #         'jitter_max': 1.4,
    #         'autoaugment_policy': 'v2',
    #         'clip_gradients_norm': 10.0,
    #         'anchor_scale': 3.0,
    #         'aspect_ratios': [1.0, 2.0, 0.5],
    #         'lr_decay_method': 'cosine'
    #     }
    # )

    # model_spec = tflite_model_maker.object_detector.EfficientDetSpec(
    #     model_name='efficientdet-lite0',
    #     uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
    #     hparams={
    #         'backbone_name': 'efficientnetv2_b3_imagenet',
    #         'nms_configs': {
    #             'method': 'gaussian',
    #             'score_thresh': 0.5,
    #             'max_nms_inputs': 5000,
    #             'max_output_size': 100
    #         },
    #         'gamma': 1.25,
    #         'label_smoothing': 0.1,
    #         'weight_decay': 8e-5,
    #         'learning_rate': 0.02,
    #         'lr_warmup_init': 0.015,
    #         'first_lr_drop_epoch': 50,
    #         'second_lr_drop_epoch': 70,
    #         'momentum': 0.9,
    #         'optimizer': 'sgd',
    #         'input_rand_hflip': True,
    #         'jitter_min': 0.6,
    #         'jitter_max': 1.4,
    #         'autoaugment_policy': 'v2',
    #         'clip_gradients_norm': 10.0,
    #         'anchor_scale': 3.0,
    #         'aspect_ratios': [1.0, 2.0, 0.5],
    #         'lr_decay_method': 'cosine'
    #     }
    # )

    #
    #     model = object_detector.ObjectDetector(model_spec, LABEL_MAP, representative_data)
    #     model.model = loaded_model

    # else:

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # spec = model_spec.get(ARCHITECTURE)

    train = object_detector.DataLoader.from_pascal_voc(
        images_dir=TRAIN_DIR,
        annotations_dir=TRAIN_DIR,
        label_map=KETTLEBELL_LABEL_MAP,
    )

    valid = object_detector.DataLoader.from_pascal_voc(
        images_dir=VALID_DIR,
        annotations_dir=VALID_DIR,
        label_map=KETTLEBELL_LABEL_MAP,

    )

    test = object_detector.DataLoader.from_pascal_voc(
        images_dir=TEST_DIR,
        annotations_dir=TEST_DIR,
        label_map=KETTLEBELL_LABEL_MAP,

    )

    print(f"Number of training images: {train.size}")
    print(f"Number of validation images: {valid.size}")
    print(f"Number of test images: {test.size}")

    print("Creating new model...")

    model = object_detector.create(
        train_data=train,
        model_spec=model_spec,
        batch_size=BATCH_SIZE,
        train_whole_model=TRAIN_WHOLE_MODEL,
        validation_data=valid,
        epochs=0,  # Set to 0 as we'll train manually
        do_train=False
    )

    # Train the model
    model = train_model(model, train, valid, EPOCHS)

    #Evaluate the model
    print("Evaluating the model...")
    metrics = model.evaluate(test, batch_size=BATCH_SIZE)
    print(f"Evaluation metrics: {metrics}")

    tflite_filename = f'{ARCHITECTURE}.tflite'

    if TRAIN_WHOLE_MODEL:
        tflite_filename = f'{ARCHITECTURE}_whole.tflite'

    print("Exporting the model...")
    model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)

    # Save the model as TensorFlow SavedModel
    # Specify the batch size for the saved model
    # pre_mode = 'infer'  # Specify the pre-processing mode
    # post_mode = 'global'  # Specify the post-processing mode, set to tflite when training the last model before conversion
    #
    # print("Model saved as Tensorflow")
    # model.export(
    #     SAVED_MODEL_PATH,
    #     batch_size=BATCH_SIZE,
    #     pre_mode=pre_mode,
    #     post_mode=post_mode
    # )

    #

""""
Example for representive data2
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
# representative_dataset = tf.data2.Dataset.from_tensor_slices(representative_image_paths)
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
