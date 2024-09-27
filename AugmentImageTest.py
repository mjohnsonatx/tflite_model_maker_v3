import math

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import os
from PIL import Image
import shutil


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.cast(image, tf.float32) / 255.0


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    boxes = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # Normalize the coordinates
        boxes.append([
            ymin / height, xmin / width,
            ymax / height, xmax / width
        ])

    return tf.constant(boxes, dtype=tf.float32)


def update_xml(xml_path, new_filename, new_boxes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update filename
    root.find('filename').text = new_filename

    # Update path
    if root.find('path') is not None:
        root.find('path').text = new_filename

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    # Update bounding boxes
    for i, obj in enumerate(root.findall('object')):
        bndbox = obj.find('bndbox')
        bndbox.find('ymin').text = str(int(new_boxes[i][0] * height))
        bndbox.find('xmin').text = str(int(new_boxes[i][1] * width))
        bndbox.find('ymax').text = str(int(new_boxes[i][2] * height))
        bndbox.find('xmax').text = str(int(new_boxes[i][3] * width))

    return tree


def random_zoom_out(image, boxes):
    original_height, original_width = tf.shape(image)[0], tf.shape(image)[1]

    # Randomly choose a zoom factor between 0.6 and 0.9 (zoom out)
    zoom_factor = tf.random.uniform([], 0.1, 0.7)

    # Calculate new dimensions
    new_height = tf.cast(tf.cast(original_height, tf.float32) * zoom_factor, tf.int32)
    new_width = tf.cast(tf.cast(original_width, tf.float32) * zoom_factor, tf.int32)

    # Resize the image (this effectively zooms out)
    zoomed_image = tf.image.resize(image, [new_height, new_width])

    # Create a blank canvas of the original size
    canvas = tf.zeros([original_height, original_width, 3], dtype=tf.float32)

    # Calculate offsets to center the zoomed image
    offset_height = (original_height - new_height) // 2
    offset_width = (original_width - new_width) // 2

    # Place the zoomed image onto the canvas
    canvas = tf.image.pad_to_bounding_box(zoomed_image, offset_height, offset_width, original_height, original_width)

    # Adjust bounding boxes
    scale_y = tf.cast(new_height, tf.float32) / tf.cast(original_height, tf.float32)
    scale_x = tf.cast(new_width, tf.float32) / tf.cast(original_width, tf.float32)
    offset_y = tf.cast(offset_height, tf.float32) / tf.cast(original_height, tf.float32)
    offset_x = tf.cast(offset_width, tf.float32) / tf.cast(original_width, tf.float32)

    boxes = tf.stack([
        boxes[:, 0] * scale_y + offset_y,
        boxes[:, 1] * scale_x + offset_x,
        boxes[:, 2] * scale_y + offset_y,
        boxes[:, 3] * scale_x + offset_x
    ], axis=-1)

    # Clip the boxes to ensure they're within [0, 1]
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)

    return canvas, boxes



def random_flip_horizontal(image, boxes):
    try:
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            boxes = tf.stack([
                boxes[:, 0],
                1 - boxes[:, 3],
                boxes[:, 2],
                1 - boxes[:, 1]
            ], axis=-1)

            # Check if any box coordinates are out of range
            if tf.reduce_any(boxes < 0.0) or tf.reduce_any(boxes > 1.0):
                raise ValueError("Bounding box out of range after flipping")

        return image, boxes
    except Exception as e:
        print(f"Error in random_flip_horizontal: {e}")
        # Return the original image and boxes if an error occurs
        return image, boxes


def augment_image(image, boxes):

    #Random brightness
    image = tf.image.random_brightness(image, max_delta=0.4)

    # Random saturation
    image = tf.image.random_saturation(image, lower=0.2, upper=1.1)

    # Random hue
    image = tf.image.random_hue(image, max_delta=0.2)

    #Random flip left-right
    image, boxes = random_flip_horizontal(image, boxes)

    #Apply random zoom out with 50% probability
    if tf.random.uniform([]) > 0.5:
        image, boxes = random_zoom_out(image, boxes)



    # Ensure the pixel values are still in [0, 1] range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, boxes


def augment_and_save(source_dir, dest_dir, num_images):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    np.random.shuffle(image_files)

    for i, image_file in enumerate(image_files[:num_images]):
        image_path = os.path.join(source_dir, image_file)
        xml_path = os.path.join(source_dir, os.path.splitext(image_file)[0] + '.xml')

        if not os.path.exists(xml_path):
            print(f"XML file not found for {image_file}, skipping...")
            continue

        # Load and augment image
        image = load_image(image_path)
        boxes = parse_xml(xml_path)
        augmented_image, augmented_boxes = augment_image(image, boxes)

        # Save augmented image
        new_filename = f"augmented_{i}_{image_file}"
        new_image_path = os.path.join(dest_dir, new_filename)
        tf.keras.preprocessing.image.save_img(new_image_path, augmented_image.numpy())

        # Update and save XML
        new_xml_tree = update_xml(xml_path, new_filename, augmented_boxes.numpy())
        new_xml_path = os.path.join(dest_dir, f"augmented_{i}_{os.path.splitext(image_file)[0]}.xml")
        new_xml_tree.write(new_xml_path)

        print(f"Saved augmented image and XML: {new_filename}")


# Example usage
if __name__ == "__main__":
    source_directory = 'Kettlebell Data/test'
    destination_directory = 'KB AUGMENT'
    number_of_images_to_augment = 1000

    augment_and_save(source_directory, destination_directory, number_of_images_to_augment)