import os
import shutil

src_dir = "data/raw/plantvillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/Train"
dst_dir = "data/raw/plantvillage/combined/"

os.makedirs(dst_dir, exist_ok=True)

for class_folder in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_folder)
    
    if not os.path.isdir(class_path):
        continue

    print(f"Processing: {class_folder}")

    dest_class_path = os.path.join(dst_dir, class_folder)
    os.makedirs(dest_class_path, exist_ok=True)

    for file in os.listdir(class_path):
        full_src = os.path.join(class_path, file)
        full_dst = os.path.join(dest_class_path, file)

        shutil.copy(full_src, full_dst)

print("\nDataset organized successfully!")
print("Combined dataset location:", dst_dir)
