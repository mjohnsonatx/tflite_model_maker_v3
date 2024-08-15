# import os
# import xml.etree.ElementTree as ET
# import cv2
# import random
#
# def parse_xml(xml_file):
#     """Parse XML file to extract bounding box coordinates."""
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#
#     boxes = []
#     for obj in root.findall('object'):
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         xmax = int(bbox.find('xmax').text)
#         ymin = int(bbox.find('ymin').text)
#         ymax = int(bbox.find('ymax').text)
#         boxes.append((xmin, ymin, xmax, ymax))
#     return boxes
#
# def draw_bounding_boxes(image_directory, num_images):
#     """Draw bounding boxes on randomly selected images as specified by their corresponding XML files."""
#     files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
#     random.shuffle(files)  # Shuffle the list of files to select randomly
#     selected_files = files[:num_images]  # Limit the number of files to process
#
#     for file in selected_files:
#         img_path = os.path.join(image_directory, file)
#         xml_path = img_path.replace('.jpg', '.xml')
#
#         # Load the image
#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"Failed to load image {img_path}")
#             continue
#
#         # Parse the corresponding XML file to get bounding boxes
#         if os.path.exists(xml_path):
#             bounding_boxes = parse_xml(xml_path)
#             for (xmin, ymin, xmax, ymax) in bounding_boxes:
#                 # Draw the bounding box
#                 cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#         else:
#             print(f"No XML file found for {img_path}")
#             continue
#
#         # Display the image
#         cv2.imshow('Image with Bounding Box', image)
#         cv2.waitKey(0)  # Wait for a key press to move to the next image
#         cv2.destroyAllWindows()
#
# # Example usage
# image_directory = 'augmented data2'
# num_images = 200  # Number of images to process
# draw_bounding_boxes(image_directory, num_images)

'''''
delete bad data2, parse through certain number of files in directory
'''

# import os
# import xml.etree.ElementTree as ET
# import cv2
# import random
#
# def parse_xml(xml_file):
#     """Parse XML file to extract bounding box coordinates."""
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#
#     boxes = []
#     for obj in root.findall('object'):
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         xmax = int(bbox.find('xmax').text)
#         ymin = int(bbox.find('ymin').text)
#         ymax = int(bbox.find('ymax').text)
#         boxes.append((xmin, ymin, xmax, ymax))
#     return boxes
#
# def draw_bounding_boxes(image_directory, num_images):
#     """Draw bounding boxes on randomly selected images as specified by their corresponding XML files."""
#     files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
#     random.shuffle(files)  # Shuffle the list of files to select randomly
#     selected_files = files[:num_images]  # Limit the number of files to process
#
#     for file in selected_files:
#         img_path = os.path.join(image_directory, file)
#         xml_path = img_path.replace('.jpg', '.xml')
#
#         # Load the image
#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"Failed to load image {img_path}")
#             continue
#
#         # Parse the corresponding XML file to get bounding boxes
#         if os.path.exists(xml_path):
#             bounding_boxes = parse_xml(xml_path)
#             for (xmin, ymin, xmax, ymax) in bounding_boxes:
#                 # Draw the bounding box
#                 cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#         else:
#             print(f"No XML file found for {img_path}")
#             continue
#
#         # Display the image
#         cv2.imshow('Image with Bounding Box', image)
#         key = cv2.waitKey(0)  # Wait for a key press to move to the next image
#         if key == ord('e'):  # If 'e' is pressed
#             os.remove(img_path)  # Remove the image file
#             os.remove(xml_path)  # Remove the XML file
#             print(f"Deleted {img_path} and {xml_path}")
#         cv2.destroyAllWindows()
#
# # Example usage
# image_directory = 'NEW DATA SPLIT/train'
# num_images = 2500  # Number of images to process
# draw_bounding_boxes(image_directory, num_images)


'''''
delete bad data2, parse through entire directory
'''


import os
import xml.etree.ElementTree as ET
import cv2


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


def draw_bounding_boxes(image_directory):

    img_num = 1

    """Draw bounding boxes on images as specified by their corresponding XML files."""
    files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
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
        print(f"Image number: {img_num}. ")
        img_num+=1
        key = cv2.waitKey(0)  # Wait for a key press to move to the next image
        if key == ord('e'):  # If 'e' is pressed
            os.remove(img_path)  # Remove the image file
            os.remove(xml_path)  # Remove the XML file
            print(f"Deleted {img_path} and {xml_path}")
        cv2.destroyAllWindows()

    print("end of directory")


# Example usage
image_directory = 'NEW DATA SPLIT/valid'
draw_bounding_boxes(image_directory)

"""
script to delete all data2 points with more than one bounding box
"""
# import os
# import xml.etree.ElementTree as ET
#
#
# def parse_xml_and_delete(xml_file, image_file):
#     """Parse XML file and delete if more than one object is detected."""
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#
#     objects = root.findall('object')
#     if len(objects) > 1:
#         os.remove(xml_file)  # Remove XML file
#         os.remove(image_file)  # Remove the corresponding image file
#         return True
#     return False
#
#
# def process_directory(directory):
#     """Process all XML and corresponding JPG files in the directory."""
#     xml_files = [f for f in os.listdir(directory) if f.endswith('.xml')]
#     total_deleted = 0
#
#     for xml_file in xml_files:
#         image_file = os.path.join(directory, xml_file.replace('.xml', '.jpg'))
#         xml_file = os.path.join(directory, xml_file)
#
#         if parse_xml_and_delete(xml_file, image_file):
#             total_deleted += 1
#             print(f"Deleted {xml_file} and {image_file}")
#
#     return total_deleted
#
#
# # Directory to process
# image_directory = 'NEW DATA SPLIT/valid'
# deleted_count = process_directory(image_directory)
# print(f"Total files deleted: {deleted_count}")
