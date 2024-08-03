import os
import cv2
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    """Parse XML file to extract bounding box coordinates."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def draw_bounding_boxes(image_directory, num_images):
    """Draw bounding boxes on images as specified by their corresponding XML files."""
    files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
    files = files[:num_images]  # Limit the number of files to process

    for file in files:
        img_path = os.path.join(image_directory, file)
        xml_path = img_path.replace('.jpg', '.xml')

        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image {img_path}")
            continue

        # Parse the corresponding XML file to get bounding boxes
        if os.path.exists(xml_path):
            bounding_boxes = parse_xml(xml_path)
            for (xmin, ymin, xmax, ymax) in bounding_boxes:
                # Draw the bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        else:
            print(f"No XML file found for {img_path}")
            continue

        # Display the image
        cv2.imshow('Image with Bounding Box', image)
        cv2.waitKey(0)  # Wait for a key press to move to the next image
        cv2.destroyAllWindows()

# Example usage
image_directory = 'augmented_images_for_training'
num_images = 15  # Number of images to process
draw_bounding_boxes(image_directory, num_images)