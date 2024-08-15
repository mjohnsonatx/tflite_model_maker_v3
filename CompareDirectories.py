import os

def count_files(directory):
    """ Count all files in the directory including files in subdirectories. """
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

def compare_directories(dir1, dir2):
    count_dir1 = count_files(dir1)
    count_dir2 = count_files(dir2)

    print(f"Total files in {dir1}: {count_dir1}")
    print(f"Total files in {dir2}: {count_dir2}")
    print(f"Difference in file count: {abs(count_dir1 - count_dir2)}")

# Specify the paths to the directories to compare
directory1 = 'NEW DATA POOL'
directory2 = 'data2/train'

compare_directories(directory1, directory2)