import os
import shutil

def combine_datasets(original_dir, augmented_dir, combined_dir):
    # Create the combined directory if it doesn't exist
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    # Copy original old data2
    for filename in os.listdir(original_dir):
        src_path = os.path.join(original_dir, filename)
        dst_path = os.path.join(combined_dir, filename)
        shutil.copy2(src_path, dst_path)
    print("Copied original old data2")

    # Copy augmented old data2
    for filename in os.listdir(augmented_dir):
        src_path = os.path.join(augmented_dir, filename)
        dst_path = os.path.join(combined_dir, filename)
        shutil.copy2(src_path, dst_path)
    print("Copied augmented old data2")

    print(f"Combined dataset created in {combined_dir}")

# Example usage
if __name__ == "__main__":
    original_directory = 'old data2/train'
    augmented_directory = 'augmented_images_for_training'
    combined_directory = 'combined training old data2'

    combine_datasets(original_directory, augmented_directory, combined_directory)