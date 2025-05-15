import os
import shutil

# 1) Adjust these paths to your setup:
LABELS_DIR = r"raw_labels"          # Folder containing .txt files
SOURCE_IMAGE_DIR = r"raw_images"    # Root folder of your original images (may have subfolders)
DEST_IMAGES_DIR = r"new_images"  # Where matched images will be copied
DEST_LABELS_DIR = r"new_labels"  # Where matched label files will be copied

# 2) Create destination dirs if they don’t exist
os.makedirs(DEST_IMAGES_DIR, exist_ok=True)
os.makedirs(DEST_LABELS_DIR, exist_ok=True)

# 3) Define which image extensions to look for
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]  # Add more if needed

def main():
    # List all .txt files in your labels directory
    label_files = [f for f in os.listdir(LABELS_DIR) if f.lower().endswith(".txt")]

    for txt_file in label_files:
        label_path = os.path.join(LABELS_DIR, txt_file)

        # Extract the base name without extension, e.g. "image1" from "image1.txt"
        base_name = os.path.splitext(txt_file)[0]

        # We'll search for something like "image1.jpg", "image1.png", etc. in the SOURCE_IMAGE_DIR
        found_image = False

        # Walk through SOURCE_IMAGE_DIR (including subfolders) to find a matching image
        for root, dirs, files in os.walk(SOURCE_IMAGE_DIR):
            for file_name in files:
                # Check if file_name base matches the label's base_name
                # and if the extension is in IMAGE_EXTS
                if os.path.splitext(file_name)[0] == base_name \
                   and os.path.splitext(file_name)[1].lower() in IMAGE_EXTS:

                    source_img_path = os.path.join(root, file_name)

                    # Copy the image to DEST_IMAGES_DIR
                    dest_img_path = os.path.join(DEST_IMAGES_DIR, file_name)
                    shutil.copy2(source_img_path, dest_img_path)

                    # Also copy the label to DEST_LABELS_DIR
                    dest_label_path = os.path.join(DEST_LABELS_DIR, txt_file)
                    shutil.copy2(label_path, dest_label_path)

                    found_image = True
                    break  # Stop searching once we’ve found & copied this image

            if found_image:
                break  # Stop walking subfolders if we already found a match

        if not found_image:
            print(f"Warning: No matching image found for label file '{txt_file}'")

    print("Done! Images with matching labels have been copied.")
    print(f"Images -> {DEST_IMAGES_DIR}")
    print(f"Labels -> {DEST_LABELS_DIR}")

if __name__ == "__main__":
    main()
