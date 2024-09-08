import os
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import shutil  # Import shutil for file copying


def process_image(file_info):
    source_dir, target_dir, filename = file_info
    image_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)

    # Process the image
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Handle common image formats
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure it has 3 channels
        image = np.array(image, dtype=np.uint8)  # Convert image to uint8
        image = Image.fromarray(image)
        image.save(target_path)

        # Handle the XML file for the image
        xml_filename = os.path.splitext(filename)[0] + '.xml'  # Change the extension to .xml
        xml_source_path = os.path.join(source_dir, xml_filename)
        xml_target_path = os.path.join(target_dir, xml_filename)

        # Check if the XML file exists and copy it
        if os.path.exists(xml_source_path):
            shutil.copy(xml_source_path, xml_target_path)
        return f"Processed {filename} and copied XML"

    return f"Skipped {filename} (unsupported format)"


def preprocess_and_save_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = [(source_dir, target_dir, filename) for filename in os.listdir(source_dir) if
             filename.lower().endswith((".png", ".jpg", ".jpeg"))]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_image, files))
        for result in results:
            print(result)  # Optional: print progress


def main():
    source_train_dir = 'NEW DATA POOL'
    target_train_dir = 'NEW DATA POOL WITH PREPROCESSING'
    preprocess_and_save_images(source_train_dir, target_train_dir)


if __name__ == '__main__':
    main()
