import os
import random
import shutil

def move_half_images_and_xmls(source_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Get a list of all .jpg files in the source directory
    jpg_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

    # Calculate 50% of the files
    num_files_to_move = len(jpg_files) // 2

    # Randomly select 50% of the files
    files_to_move = random.sample(jpg_files, num_files_to_move)

    for jpg_file in files_to_move:
        # Move the .jpg file
        shutil.move(os.path.join(source_dir, jpg_file), os.path.join(dest_dir, jpg_file))

        # Construct the corresponding .xml file name
        xml_file = jpg_file.replace('.jpg', '.xml')

        # Move the .xml file if it exists
        if os.path.exists(os.path.join(source_dir, xml_file)):
            shutil.move(os.path.join(source_dir, xml_file), os.path.join(dest_dir, xml_file))

    print(f"Moved {num_files_to_move} .jpg files and their associated .xml files to {dest_dir}")

# Example usage
source_directory = "NEW DATA SPLIT/test"
destination_directory = "NEW DATA SPLIT/train"

move_half_images_and_xmls(source_directory, destination_directory)
