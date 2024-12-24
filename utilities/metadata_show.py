from PIL import Image
from PIL.PngImagePlugin import PngInfo
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
import os

# Set the file to scan
FILE_TO_SCAN = "ComfyUI/..."

def read_metadata_with_pillow(file_path):
    try:
        with Image.open(file_path) as img:
            img.load()  # Load image to ensure info is populated
            metadata = img.info
            if metadata:
                print(f"Metadata from Pillow for {file_path}:")
                for key, value in metadata.items():
                    print(f"{key}: {value}")
            else:
                print(f"No Pillow metadata found for {file_path}.")
    except Exception as e:
        print(f"Cannot read {file_path} with Pillow: {e}")

def read_metadata_with_hachoir(file_path):
    try:
        parser = createParser(file_path)
        metadata = extractMetadata(parser)
        if metadata:
            print(f"Metadata from Hachoir for {file_path}:")
            for line in metadata.exportPlaintext():
                print(line)
        else:
            print(f"No Hachoir metadata found for {file_path}.")
    except Exception as e:
        print(f"Cannot read {file_path} with Hachoir: {e}")

if __name__ == "__main__":
    if os.path.exists(FILE_TO_SCAN):
        print(f"File found: {FILE_TO_SCAN}")
        read_metadata_with_pillow(FILE_TO_SCAN)
        print("")
        read_metadata_with_hachoir(FILE_TO_SCAN)
    else:
        print(f"File '{FILE_TO_SCAN}' not found.")