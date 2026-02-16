from __future__ import annotations
import os
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT_DIR = Path("BB_bench_press_extracted_frames_combined/annotation_plus_augmentation")


# TODO need to build filter the filter out the kb data that does not have xmls
def update_xml_files_in_place(root_dir):
    # subdirectories = ['train', 'test', 'valid']
    #
    # for sub in subdirectories:
    #     input_dir = os.path.join(root_dir, sub)

        # Process each file in the subdirectory
        for file in os.listdir(root_dir):
            if file.endswith('.xml'):
                xml_path = os.path.join(root_dir, file)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Flag to track if XML is modified
                modified = False

                # Check each object for unwanted tags and incorrect names
                for obj in root.findall('object'):
                    name = obj.find('name')
                    # Update object name if necessary
                    if name.text == 'kettlebell':
                        name.text = 'barbell'
                        modified = True

                    # Remove any polygon tags found
                    polygon = obj.find('polygon')
                    if polygon is not None:
                        obj.remove(polygon)
                        modified = True

                # If modifications were made, overwrite the existing XML file
                if modified:
                    tree.write(xml_path)
                    print(f"Updated in place: {xml_path}")
                else:
                    print(f"No updates needed for: {xml_path}")

update_xml_files_in_place(ROOT_DIR)