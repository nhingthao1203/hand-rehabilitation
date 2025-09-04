import os
import shutil

# Đường dẫn chứa các thư mục cần gộp
parent_dir = r"D:\gesture_recognition\dataset_split"

# Danh sách các thư mục chính chứa dữ liệu
source_folders = [
     "processed_dataNhi","val"
]

# Thư mục đích chứa dữ liệu gộp
final_dataset = os.path.join(parent_dir, "val_final_dataset")
os.makedirs(final_dataset, exist_ok=True)

# Các loại cử chỉ tay cần gộp
gesture_types = ["1f_hand", "2f_hand", "3f_hand", "4f_hand",
                 "close_hand", "duck_hand", "open_hand", "spread_hand"]

# Tạo thư mục con cho từng cử chỉ tay trong final_dataset
for gesture in gesture_types:
    os.makedirs(os.path.join(final_dataset, gesture), exist_ok=True)

# Duyệt qua từng thư mục nguồn và gộp ảnh
for folder in source_folders:
    source_path = os.path.join(parent_dir, folder)

    if not os.path.exists(source_path):  # Kiểm tra nếu thư mục tồn tại
        print(f"⚠️ Bỏ qua {folder} (không tồn tại)")
        continue

    print(f"📂 Đang xử lý: {folder}")

    for gesture in gesture_types:
        gesture_source = os.path.join(source_path, gesture)  # Thư mục chứa ảnh
        gesture_dest = os.path.join(final_dataset, gesture)  # Thư mục đích

        if os.path.exists(gesture_source):
            # Duyệt qua từng ảnh trong thư mục gesture_source
            for img_name in os.listdir(gesture_source):
                img_source_path = os.path.join(gesture_source, img_name)
                img_dest_path = os.path.join(gesture_dest, img_name)

                # Đảm bảo không bị trùng tên file
                base_name, ext = os.path.splitext(img_name)
                counter = 1
                while os.path.exists(img_dest_path):
                    img_dest_path = os.path.join(gesture_dest, f"{base_name}_{counter}{ext}")
                    counter += 1

                # Di chuyển hoặc sao chép ảnh
                shutil.move(img_source_path, img_dest_path)  # Chuyển ảnh (hoặc dùng shutil.copy để sao chép)

        print(f"✅ Hoàn tất gộp {gesture} từ {folder}.")

print("🎯 Hoàn tất! Dữ liệu đã được gộp vào thư mục 'final_dataset'.")
