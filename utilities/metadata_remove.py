from PIL import Image
import os

# Set the file to clean
FILE_TO_CLEAN = "ComfyUI/"

def show_metadata(file_path):
    """ Display the metadata of the image file. """
    print(f"\nMetadata for {file_path}:")
    try:
        with Image.open(file_path) as img:
            img.load()  # Ensuring info is populated for PNG images
            metadata = img.info
            if metadata:
                for key, value in metadata.items():
                    print(f"{key}: {value}")
            else:
                print("No metadata found.")
    except Exception as e:
        print(f"Cannot read {file_path}: {e}")

def remove_metadata_in_place(file_path):
    """ Remove all metadata from the image file and save it in place. """
    try:
        with Image.open(file_path) as img:
            data = img.copy()
            # Save image without any metadata
            data.save(file_path, "PNG")
            print(f"\nMetadata removed from {file_path}")
    except Exception as e:
        print(f"Cannot remove metadata from {file_path}: {e}")

if __name__ == "__main__":
    if os.path.exists(FILE_TO_CLEAN):
        print(f"File found: {FILE_TO_CLEAN}")
        
        # Show current metadata
        show_metadata(FILE_TO_CLEAN)
        
        # Remove metadata
        remove_metadata_in_place(FILE_TO_CLEAN)
        
        # Show metadata after cleaning
        show_metadata(FILE_TO_CLEAN)
    else:
        print(f"File '{FILE_TO_CLEAN}' not found.")