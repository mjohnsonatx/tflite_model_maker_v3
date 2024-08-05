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


def rotate_image(image, angle):
    """Rotate an image using TensorFlow operations."""
    # Convert angle to radians
    angle = tf.cast(angle * np.pi / 180, tf.float32)

    # Get image shape
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    # Calculate new image size
    cos_angle = tf.math.cos(angle)
    sin_angle = tf.math.sin(angle)
    new_h = tf.cast(h * tf.abs(cos_angle) + w * tf.abs(sin_angle), tf.int32)
    new_w = tf.cast(w * tf.abs(cos_angle) + h * tf.abs(sin_angle), tf.int32)

    # Rotation matrix
    rotation_matrix = tf.stack([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # Rotate image
    center = tf.stack([w / 2, h / 2])
    grid_x, grid_y = tf.meshgrid(tf.range(new_w, dtype=tf.float32), tf.range(new_h, dtype=tf.float32))
    grid = tf.stack([grid_x, grid_y], axis=-1)

    grid -= tf.cast(tf.stack([new_w / 2, new_h / 2]), tf.float32)
    rotated_grid = tf.einsum('ij,hwj->hwi', rotation_matrix, grid)
    rotated_grid += center

    # Interpolate
    x = rotated_grid[..., 0]
    y = rotated_grid[..., 1]
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, tf.cast(w - 1, tf.int32))
    x1 = tf.clip_by_value(x1, 0, tf.cast(w - 1, tf.int32))
    y0 = tf.clip_by_value(y0, 0, tf.cast(h - 1, tf.int32))
    y1 = tf.clip_by_value(y1, 0, tf.cast(h - 1, tf.int32))

    Ia = tf.gather_nd(image, tf.stack([y0, x0], -1))
    Ib = tf.gather_nd(image, tf.stack([y0, x1], -1))
    Ic = tf.gather_nd(image, tf.stack([y1, x0], -1))
    Id = tf.gather_nd(image, tf.stack([y1, x1], -1))

    wa = (tf.cast(x1, tf.float32) - x) * (tf.cast(y1, tf.float32) - y)
    wb = (x - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - y)
    wc = (tf.cast(x1, tf.float32) - x) * (y - tf.cast(y0, tf.float32))
    wd = (x - tf.cast(x0, tf.float32)) * (y - tf.cast(y0, tf.float32))

    rotated_image = tf.add_n([wa[..., tf.newaxis] * Ia,
                              wb[..., tf.newaxis] * Ib,
                              wc[..., tf.newaxis] * Ic,
                              wd[..., tf.newaxis] * Id])

    return rotated_image


def rotate_box(box, angle_degrees):
    ymin, xmin, ymax, xmax = tf.unstack(box)

    # Center of box
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    # Box dimensions
    width = xmax - xmin
    height = ymax - ymin

    # Convert angle to radians
    angle_rad = tf.cast(angle_degrees * np.pi / 180.0, dtype=tf.float32)

    cos_angle = tf.cos(angle_rad)
    sin_angle = tf.sin(angle_rad)

    # New dimensions
    new_width = abs(width * cos_angle) + abs(height * sin_angle)
    new_height = abs(height * cos_angle) + abs(width * sin_angle)

    # New coordinates
    new_xmin = tf.clip_by_value(center_x - new_width / 2.0, 0.0, 1.0)
    new_ymin = tf.clip_by_value(center_y - new_height / 2.0, 0.0, 1.0)
    new_xmax = tf.clip_by_value(center_x + new_width / 2.0, 0.0, 1.0)
    new_ymax = tf.clip_by_value(center_y + new_height / 2.0, 0.0, 1.0)

    # Diagnostics
    before = f"Pre-rotation: {xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}"
    after = f"Post-rotation: {new_xmin:.4f}, {new_ymin:.4f}, {new_xmax:.4f}, {new_ymax:.4f}"
    print(f"Angle: {angle_degrees:.2f}°: {before} → {after}")

    # Safety checks
    if new_xmin >= new_xmax or new_ymin >= new_ymax:
        print(f"WARNING: Invalid box after rotation: {new_xmin:.4f}, {new_ymin:.4f}, {new_xmax:.4f}, {new_ymax:.4f}")
        return box  # Return original if rotation produces invalid box

    return tf.stack([new_ymin, new_xmin, new_ymax, new_xmax])


def rotate_image_and_boxes(image, boxes, max_angle=15):
    angle = tf.random.uniform([], -max_angle, max_angle)

    # Rotate image
    image = rotate_image(image, angle)

    # Get image dimensions
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)

    # Rotate each box
    rotated_boxes = tf.stack([rotate_box(box, angle) for box in boxes])

    return image, rotated_boxes



def augment_image(image, boxes):

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.5)

    # Random saturation
    image = tf.image.random_saturation(image, lower=0.2, upper=1.1)

    # Random hue
    image = tf.image.random_hue(image, max_delta=0.2)

    # Random flip left-right
    image, boxes = random_flip_horizontal(image, boxes)

    # if tf.random.uniform([]) < 0.99:
    #     image, boxes = rotate_image_and_boxes(image, boxes)



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
    source_directory = 'data/train'
    destination_directory = 'augmented data with added rotation'
    number_of_images_to_augment = 15

    augment_and_save(source_directory, destination_directory, number_of_images_to_augment)