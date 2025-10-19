import os
import glob
import logging
from xml.etree import ElementTree as ET

#TODO add this feature/function to the DataCleaningTab or FileOperationsTab in the GUIModelMaker


# --- CONFIGURE THESE ---
DATA_DIR = r'FINAL_KB_DATA'
FOLDERS = ['train', 'valid', 'test']  # subfolders to process
# ------------------------

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# ----------------------

def pick_existing_image(images_dir, basename_no_ext):
    """Return image filename (with extension) if found, else None."""
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        path = os.path.join(images_dir, basename_no_ext + ext)
        if os.path.isfile(path):
            return os.path.basename(path)
    return None


def fix_directory(data_dir, folder):
    """Fix <filename> and <path> fields in XML files for a given folder."""
    dir_path = os.path.join(data_dir, folder)
    images_dir = dir_path  # adjust if images live elsewhere
    xml_files = glob.glob(os.path.join(dir_path, '*.xml'))

    logger.info(f"🔍 Processing folder: {folder} ({len(xml_files)} XML files found)")

    fixed, skipped = 0, 0

    for idx, xml_path in enumerate(xml_files, start=1):
        base = os.path.splitext(os.path.basename(xml_path))[0]
        img_name = pick_existing_image(images_dir, base)

        if not img_name:
            logger.warning(f"[{folder}] ⚠️ No image found for {base}")
            skipped += 1
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Update <filename>
            fn = root.find('filename')
            if fn is None:
                fn = ET.SubElement(root, 'filename')
            old_filename = fn.text
            fn.text = img_name

            # Update <path>
            p = root.find('path')
            if p is None:
                p = ET.SubElement(root, 'path')
            old_path = p.text
            new_path = os.path.abspath(os.path.join(images_dir, img_name))
            p.text = new_path

            # Write back
            tree.write(xml_path, encoding='utf-8', xml_declaration=False)
            fixed += 1

            logger.info(
                f"[{folder}] ✅ Fixed {idx}/{len(xml_files)}: "
                f"{os.path.basename(xml_path)} "
                f"(filename: '{old_filename}' → '{img_name}')"
            )

        except Exception as e:
            logger.error(f"[{folder}] ❌ Error fixing {xml_path}: {e}")
            skipped += 1

    logger.info(
        f"📁 Done with folder '{folder}': Fixed={fixed}, Skipped={skipped}\n"
        + "-" * 80
    )


if __name__ == "__main__":
    logger.info("🚀 Starting XML filename/path synchronization...")
    # for folder in FOLDERS:
    #     fix_directory(DATA_DIR, folder)

    fix_directory(DATA_DIR, 'train' )
    fix_directory(DATA_DIR, 'valid')
    logger.info("✅ All folders processed successfully.")
