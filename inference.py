import os
import cv2
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})

    return objects


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(image, input_size):
    image = tf.image.resize(image, input_size)
    image = tf.cast(image, tf.uint8)  # Cast the image to float32 for quant model, unit8 for non-quant
    return image


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='old_models/efficientdet_lite0_whole_b2e50_original_data_augmented.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shapes:")
for output_detail in output_details:
    print(output_detail['name'], output_detail['shape'])

input_shape = input_details[0]['shape']
input_size = input_shape[1:3]

# Iterate through the test old data
test_dir = 'augmented_images_for_training'
original_test = 'old data/test'
output_dir = 'test_lite0_b2_e50_original_data_with_augment'  # Directory to save the output images
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

"""
use this version for non-quantized models
"""
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(test_dir, filename)
        xml_path = os.path.join(test_dir, filename[:-4] + '.xml')

        # Load and preprocess the image
        image = load_image(image_path)
        input_tensor = preprocess_image(image, input_size)
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_tensor)

        # Run inference
        interpreter.invoke()

        # Get the output tensors
        boxes = interpreter.get_tensor(output_details[1]['index'])
        classes = interpreter.get_tensor(output_details[3]['index'])
        scores = interpreter.get_tensor(output_details[0]['index'])
        num_detections = interpreter.get_tensor(output_details[2]['index'])

        print("num_detections shape:", num_detections.shape)

        # Convert num_detections to an integer
        num_detections = int(num_detections.item())

        # Parse the XML file to get ground truth annotations
        # ground_truth = parse_xml(xml_path)

        # Find the detection with the highest score
        max_score_index = np.argmax(scores[0][:num_detections])

        # Draw bounding box for the detection with the highest score
        ymin, xmin, ymax, xmax = boxes[0][max_score_index]
        xmin = int(xmin * image.shape[1])
        ymin = int(ymin * image.shape[0])
        xmax = int(xmax * image.shape[1])
        ymax = int(ymax * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Save the image with bounding box to a file
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)

        print(f"Processed {filename}")

"""
Use for quantized models
"""
# for filename in os.listdir(test_dir):
#     if filename.endswith('.jpg'):
#         image_path = os.path.join(test_dir, filename)
#         xml_path = os.path.join(test_dir, filename[:-4] + '.xml')
#
#         # Load and preprocess the image
#         image = load_image(image_path)
#         input_tensor = preprocess_image(image, input_size)
#         input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension
#
#         # Set the input tensor
#         interpreter.set_tensor(input_details[0]['index'], input_tensor)
#
#         # Run inference
#         interpreter.invoke()
#
#         # Get the output tensors
#         boxes = interpreter.get_tensor(output_details[1]['index'])
#         classes = interpreter.get_tensor(output_details[3]['index'])
#         scores = interpreter.get_tensor(output_details[0]['index'])
#
#         # Reshape the output tensors
#         boxes = np.squeeze(boxes)
#         classes = np.squeeze(classes)
#         scores = np.squeeze(scores)
#
#         # Parse the XML file to get ground truth annotations
#         ground_truth = parse_xml(xml_path)
#
#         # Find the detections with scores above a threshold
#         threshold = 0.5
#         detections = np.where(scores > threshold)
#
#         # Draw bounding boxes for the detections
#         for detection in zip(*detections):
#             class_id, ymin, xmin, ymax, xmax = detection
#             xmin = int(xmin * image.shape[1])
#             ymin = int(ymin * image.shape[0])
#             xmax = int(xmax * image.shape[1])
#             ymax = int(ymax * image.shape[0])
#             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#
#         # Save the image with bounding boxes to a file
#         output_path = os.path.join(output_dir, filename)
#         cv2.imwrite(output_path, image)
#
#         print(f"Processed {filename}")