import os
import random
import shutil
from collections import defaultdict

# --------- USER SETTINGS -----------
SOURCE_IMAGES = r"new_images"   # Where your labeled images are
SOURCE_LABELS = r"new_labels"   # Where your label .txt files are
DEST_DATASET  = r"datasets\my_dataset"  # Final YOLO directory

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.2

# 1) Make output directories
os.makedirs(os.path.join(DEST_DATASET, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(DEST_DATASET, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(DEST_DATASET, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(DEST_DATASET, "labels", "val"), exist_ok=True)

# 2) Gather all images & parse classes
all_images = []
for file_name in os.listdir(SOURCE_IMAGES):
    f_lower = file_name.lower()
    if f_lower.endswith((".jpg", ".jpeg", ".png")):
        base = os.path.splitext(file_name)[0]
        label_path = os.path.join(SOURCE_LABELS, base + ".txt")
        if not os.path.exists(label_path):
            print(f"Warning: no .txt label for {file_name}, skipping...")
            continue

        class_ids = set()
        with open(label_path, "r") as lf:
            for line in lf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])
                class_ids.add(class_id)


        all_images.append({
            "filename": file_name,
            "class_ids": class_ids
        })

class_to_imgs = defaultdict(list)
for i, img_info in enumerate(all_images):
    for c in img_info["class_ids"]:
        class_to_imgs[c].append(i)

assignment = [None] * len(all_images)


classes_list = list(class_to_imgs.keys())
random.shuffle(classes_list)

for c in classes_list:
    img_indices = class_to_imgs[c]
    random.shuffle(img_indices)

    n = len(img_indices)
    if n == 0:
        continue


    train_target = int(n * TRAIN_RATIO)
    val_target   = n - train_target

    used_train = 0
    used_val   = 0

    for idx in img_indices:
        if assignment[idx] is not None:

            if assignment[idx] == "train":
                used_train += 1
            elif assignment[idx] == "val":
                used_val += 1
            continue


        if used_train < train_target:
            assignment[idx] = "train"
            used_train += 1
        else:
            assignment[idx] = "val"
            used_val += 1


for i, subset in enumerate(assignment):
    if subset is None:
        assignment[i] = "train"

for i, img_info in enumerate(all_images):
    subset = assignment[i]
    img_file = img_info["filename"]
    base = os.path.splitext(img_file)[0]

    src_img = os.path.join(SOURCE_IMAGES, img_file)
    src_lbl = os.path.join(SOURCE_LABELS, base + ".txt")

    dst_img = os.path.join(DEST_DATASET, "images", subset, img_file)
    dst_lbl = os.path.join(DEST_DATASET, "labels", subset, base + ".txt")

    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)

print("Train/Val stratified split complete!")
print(f"Check {DEST_DATASET}/images/train, {DEST_DATASET}/images/val, etc.")
