import os

# Thay bằng đường dẫn thư mục chứa các thư mục chính (train, val, test)
parent_dir = r"D:\gesture_recognition\dataset_split"  # Cập nhật đường dẫn thực tế của bạn

# Các định dạng file ảnh hợp lệ
valid_extensions = {".jpg", ".jpeg", ".png"}

# Dictionary để lưu số lượng ảnh của mỗi thư mục con
image_counts = {}

# Duyệt qua từng thư mục chính (train, val, test)
for folder in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder)

    if os.path.isdir(folder_path):  # Chỉ xét thư mục chính
        print(f"\n📂 {folder}:")  # In tên thư mục chính

        total_images = 0  # Biến đếm tổng số ảnh trong thư mục chính

        # Duyệt từng thư mục con (các loại cử chỉ tay)
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)

            if os.path.isdir(subfolder_path):  # Đảm bảo là thư mục con
                # Đếm số ảnh trong thư mục con này
                num_images = sum(
                    1 for root, _, files in os.walk(subfolder_path)
                    for file in files if os.path.splitext(file)[1].lower() in valid_extensions
                )

                # Cộng dồn vào tổng số ảnh của thư mục chính
                total_images += num_images

                # Lưu kết quả từng thư mục con
                image_counts[subfolder_path] = num_images
                print(f"   📁 {subfolder}: {num_images} hình ảnh")

        # In tổng số ảnh trong thư mục chính
        print(f"\n📂 Tổng số ảnh trong {folder}: {total_images} hình ảnh\n")

# Nếu muốn lưu kết quả vào file, bạn có thể mở một file và ghi vào:
with open("image_count_results.txt", "w", encoding="utf-8") as f:
    for folder, count in image_counts.items():
        f.write(f"{folder}: {count} hình ảnh\n")

print("\n✅ Hoàn tất! Kết quả đã in ra màn hình và lưu vào 'image_count_results.txt'.")
