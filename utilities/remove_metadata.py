
# utilities/remove_metadata.py
import os
import logging
from PIL import Image
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def show_metadata(file_path):
    """ Display the metadata of the image file. """
    logging.info(f"Metadata for {file_path}:")
    try:
        with Image.open(file_path) as img:
            img.load()  # Ensuring info is populated for PNG images
            metadata = img.info
            if metadata:
                for key, value in metadata.items():
                    logging.info(f"{key}: {value}")
            else:
                logging.info("No metadata found.")
    except Exception as e:
        logging.warning(f"Cannot read {file_path}: {e}")

def has_metadata(file_path):
    """ Check if the image file has metadata. """
    try:
        with Image.open(file_path) as img:
            img.load()  # Ensuring info is populated for PNG images
            return bool(img.info)
    except Exception as e:
        logging.warning(f"Cannot read {file_path}: {e}")
        return False

def remove_metadata_in_place(file_path):
    """ Remove all metadata from the image file and save it in place. """
    try:
        with Image.open(file_path) as img:
            data = img.copy()
            # Save image without any metadata
            data.save(file_path, "PNG")
            logging.info(f"Metadata removed from {file_path}")
            return True
    except Exception as e:
        logging.warning(f"Cannot remove metadata from {file_path}: {e}")
        return False

def remove_metadata_from_all_images(directory):
    file_count = 0
    files_with_metadata = 0
    removal_count = 0
    start_time = time.time()
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                file_count += 1
                file_path = os.path.join(root, file)
                
                if has_metadata(file_path):
                    files_with_metadata += 1

                    # Show current metadata
                    show_metadata(file_path)
                    
                    # Remove metadata
                    if remove_metadata_in_place(file_path):
                        removal_count += 1
                    
                    # Show metadata after cleaning
                    show_metadata(file_path)
    
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"\nMetadata removal completed in {duration:.2f} seconds.")
    logging.info(f"Total files processed: {file_count}")
    logging.info(f"Files with metadata: {files_with_metadata}")
    logging.info(f"Metadata removed from {removal_count} files")

if __name__ == "__main__":
    remove_metadata_from_all_images(".")