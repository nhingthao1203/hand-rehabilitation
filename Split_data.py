import os
import shutil
import random

# Định nghĩa thư mục nguồn và thư mục đích
source_dir = r"D:\gesture_recognition\processed\final_dataset"  # Thư mục chứa ảnh gốc sau xử lý
output_dir = r"D:\gesture_recognition\dataset_split"  # Thư mục chứa tập train, val, test

# Định nghĩa tỷ lệ chia
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Định dạng dữ liệu đầu ra
sets = ["train", "val", "test"]
for s in sets:
    for category in os.listdir(source_dir):
        os.makedirs(os.path.join(output_dir, s, category), exist_ok=True)

# Chia dữ liệu
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)

    if os.path.isdir(category_path):
        images = [img for img in os.listdir(category_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        # Tính số lượng ảnh cho từng tập
        num_train = int(len(images) * train_ratio)
        num_val = int(len(images) * val_ratio)

        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # Sao chép ảnh vào thư mục tương ứng
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "train", category, img))

        for img in val_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "val", category, img))

        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "test", category, img))

print("✅ Hoàn tất chia dữ liệu!")
