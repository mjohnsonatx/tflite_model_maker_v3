import os
import shutil

def combine_datasets(original_dir, augmented_dir, combined_dir):
    # Create the combined directory if it doesn't exist
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    # Copy original old data
    for filename in os.listdir(original_dir):
        src_path = os.path.join(original_dir, filename)
        dst_path = os.path.join(combined_dir, filename)
        shutil.copy2(src_path, dst_path)
    print("Copied original old data")

    # Copy augmented old data
    for filename in os.listdir(augmented_dir):
        src_path = os.path.join(augmented_dir, filename)
        dst_path = os.path.join(combined_dir, filename)
        shutil.copy2(src_path, dst_path)
    print("Copied augmented old data")

    print(f"Combined dataset created in {combined_dir}")

# Example usage
if __name__ == "__main__":
    original_directory = 'old data/train'
    augmented_directory = 'augmented_images_for_training'
    combined_directory = 'combined training old data'

    combine_datasets(original_directory, augmented_directory, combined_directory)