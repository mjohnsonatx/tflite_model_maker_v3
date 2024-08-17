# from tflite_model_maker import object_detector
# from tflite_model_maker import model_spec
# import os
#
# # Disable training on GPU.
# # tf.config.set_visible_devices([], 'GPU')
#
# EXPORT_DIR = 'models'
# DATA_DIR = 'NEW DATA SPLIT'
# TRAIN_DIR = os.path.join(DATA_DIR, 'train')
# VALID_DIR = os.path.join(DATA_DIR, 'valid')
# TEST_DIR = os.path.join(DATA_DIR, 'test')
# BATCH_SIZE = 4
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
#         label_map={1: "barbell"}
#     )
#
#     valid = object_detector.DataLoader.from_pascal_voc(
#         images_dir=VALID_DIR,
#         annotations_dir=VALID_DIR,
#         label_map={1: "barbell"}
#     )
#
#     test = object_detector.DataLoader.from_pascal_voc(
#         images_dir=TEST_DIR,
#         annotations_dir=TEST_DIR,
#         label_map={1: "barbell"}
#     )
#
#     model = object_detector.create(
#         train,
#         epochs=50,
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
#     print("Evaluating the original model...")
#     print(model.evaluate(test, batch_size=BATCH_SIZE))
#
#     print("Exporting the model...")
#     model.export(export_dir=EXPORT_DIR, tflite_filename=tflite_filename)
#
#     print("Evaluating the exported model...")
#     print(model.evaluate_tflite(os.path.join(EXPORT_DIR, tflite_filename), test))


import logging
import os

import hub
import tensorflow as tf
import tflite_model_maker
from tflite_model_maker import object_detector, model_spec

EXPORT_DIR = 'models'
DATA_DIR = 'NEW DATA SPLIT'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
BATCH_SIZE = 4
EPOCHS = 65
BACKBONE = 'efficientnetv2_b1_imagenet'
ARCHITECTURE = 'efficientdet_lite0'
TRAIN_WHOLE_MODEL = True

LABEL_MAP = {1: "barbell"}


def train_model(model, train_data, validation_data, epochs):
    model.train(
        train_data=train_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=BATCH_SIZE
    )
    return model


if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)

    spec = model_spec.get(ARCHITECTURE)

    train = object_detector.DataLoader.from_pascal_voc(
        images_dir=TRAIN_DIR,
        annotations_dir=TRAIN_DIR,
        label_map=LABEL_MAP
    )

    valid = object_detector.DataLoader.from_pascal_voc(
        images_dir=VALID_DIR,
        annotations_dir=VALID_DIR,
        label_map=LABEL_MAP
    )

    test = object_detector.DataLoader.from_pascal_voc(
        images_dir=TEST_DIR,
        annotations_dir=TEST_DIR,
        label_map=LABEL_MAP
    )

    # representative_data = object_detector.DataLoader.from_pascal_voc(
    #     images_dir='representative data2',
    #     annotations_dir='representative data2',
    #     label_map=LABEL_MAP
    # )

    print(f"Number of training images: {train.size}")
    print(f"Number of validation images: {valid.size}")
    print(f"Number of test images: {test.size}")

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
                'iou_thresh': None,
                'score_thresh': 0.5,
                'sigma': None,
                'pyfunc': False,
                'max_nms_inputs': 0,
                'max_output_size': 100},
            'gamma': 1.25,
            'label_smoothing': 0.1
        }
    )

    #
    #     model = object_detector.ObjectDetector(model_spec, LABEL_MAP, representative_data)
    #     model.model = loaded_model

    # else:
    print("Creating new model...")
    # model = object_detector.create(
    #     train,
    #     epochs=50,
    #     model_spec=spec,
    #     batch_size=BATCH_SIZE,
    #     train_whole_model=TRAIN_WHOLE_MODEL,
    #     validation_data=valid
    # )

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
    # # Export quantized TFLite model
    # model.export(export_dir=EXPORT_DIR,
    #              tflite_filename=f'{ARCHITECTURE}_retrained_quantized.tflite',
    #              quantization_config=model.get_default_quantization_config(representative_data=test))
    # print(f"Quantized model exported as TFLite to {EXPORT_DIR}/{ARCHITECTURE}_retrained_quantized.tflite")

    # model = object_detector.create(
    #     train_data=train,
    #     epochs = 30,
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
