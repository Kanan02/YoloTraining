import os
import shutil

# --- EDIT THESE PATHS ---
RAW_IMAGES_DIR = r"raw_images"      # your raw images root (with subfolders)
USED_IMAGES_DIR = r"new_images"     # the images you've already used/annotated
UNUSED_IMAGES_DIR = r"unused_images" # where to copy images that aren't used yet

# Supported image extensions (feel free to add more)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def main():
    # 1) Gather all filenames in USED_IMAGES_DIR (the ones already used)
    used_filenames = set()

    # new_images might have subfolders or might be flat – adjust if needed
    # If your new_images folder is flat, we can just listdir
    for f in os.listdir(USED_IMAGES_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTS:
            used_filenames.add(f.lower())  # store the lowercase name

    # 2) Create the unused_images directory if not exist
    os.makedirs(UNUSED_IMAGES_DIR, exist_ok=True)

    # 3) Walk through RAW_IMAGES_DIR recursively
    for root, dirs, files in os.walk(RAW_IMAGES_DIR):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in IMAGE_EXTS:
                # Check if the file is in used_filenames
                if file_name.lower() not in used_filenames:
                    # This means we haven't used this image yet
                    src_path = os.path.join(root, file_name)
                    dst_path = os.path.join(UNUSED_IMAGES_DIR, file_name)

                    # Copy it
                    shutil.copy2(src_path, dst_path)

    print("All unused images have been copied to:", UNUSED_IMAGES_DIR)

if __name__ == "__main__":
    main()
