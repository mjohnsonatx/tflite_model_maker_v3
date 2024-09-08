import os
import shutil
import numpy as np


def split_data(source_dir, train_dir, test_dir, valid_dir, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    # Get all JPEG files
    files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
    np.random.shuffle(files)  # Shuffle to randomize the distribution

    # Calculate split indices
    total_files = len(files)
    train_end = int(train_ratio * total_files)
    test_end = train_end + int(test_ratio * total_files)

    # Split files into train, test, and validation sets
    train_files = files[:train_end]
    test_files = files[train_end:test_end]
    valid_files = files[test_end:]

    # Function to copy files
    def copy_files(files, destination):
        for file in files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(destination, file))
            xml_file = os.path.splitext(file)[0] + '.xml'
            shutil.copy(os.path.join(source_dir, xml_file), os.path.join(destination, xml_file))

    # Copy files to corresponding directories
    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)
    copy_files(valid_files, valid_dir)

    print(f"Files distributed: {len(train_files)} train, {len(test_files)} test, {len(valid_files)} valid")


# Define directories
source_directory = 'NEW DATA POOL WITH PREPROCESSING'
train_directory = 'NEW PREPROCESSED DATA/train'
test_directory = 'NEW PREPROCESSED DATA/test'
valid_directory = 'NEW PREPROCESSED DATA/valid'

# Call the function
split_data(source_directory, train_directory, test_directory, valid_directory)
