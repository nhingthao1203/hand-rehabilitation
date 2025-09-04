import cv2
import os
import numpy as np
import mediapipe as mp

# Khởi tạo Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Khởi tạo model phát hiện bàn tay
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Định nghĩa thư mục đầu vào và đầu ra
data_dir = r"D:\gesture_recognition\collected\dataNhi"
output_dir = "dataset_split/processed_dataNhi"
os.makedirs(output_dir, exist_ok=True)

# Thông số tiền xử lý
offset = 20
crop_size = (224, 224)


# Hàm cân bằng sáng/tương phản
def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)


# Duyệt qua từng thư mục con trong data_dir
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)

    if os.path.isdir(category_path):
        print(f"🔹 Đang xử lý thư mục: {category}")

        # Tạo thư mục đầu ra tương ứng
        category_output_path = os.path.join(output_dir, category)
        os.makedirs(category_output_path, exist_ok=True)

        # Duyệt qua từng file ảnh trong thư mục
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Không thể đọc ảnh: {img_path}")
                continue

            img = enhance_contrast(img)  # Cân bằng độ tương phản

            # Chuyển ảnh sang không gian màu RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Phát hiện bàn tay
            results = hands.process(img_rgb)

            # Kiểm tra nếu phát hiện bàn tay
            if not results.multi_hand_landmarks:
                print(f"❌ Không phát hiện bàn tay: {img_name}")
                continue

            print(f"✅ Bàn tay được phát hiện trong ảnh: {img_name}")

            # Vẽ landmarks lên ảnh
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy tọa độ bàn tay
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = img.shape

            x_min = min([int(lm.x * w) for lm in hand_landmarks.landmark])
            y_min = min([int(lm.y * h) for lm in hand_landmarks.landmark])
            x_max = max([int(lm.x * w) for lm in hand_landmarks.landmark])
            y_max = max([int(lm.y * h) for lm in hand_landmarks.landmark])

            # Mở rộng vùng cắt với offset
            x1, y1 = max(0, x_min - offset), max(0, y_min - offset)
            x2, y2 = min(w, x_max + offset), min(h, y_max + offset)

            # Kiểm tra kích thước vùng cắt
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                print(f"❌ Vùng cắt quá nhỏ: {img_name}")
                continue

            # Cắt ảnh bàn tay
            imgCrop = img[y1:y2, x1:x2]

            # Resize về kích thước chuẩn (224x224)
            imgCrop = cv2.resize(imgCrop, crop_size, interpolation=cv2.INTER_AREA)

            # Lưu ảnh đã xử lý vào thư mục mới
            save_path = os.path.join(category_output_path, img_name)
            cv2.imwrite(save_path, imgCrop)
            print(f"📁 Ảnh đã lưu: {save_path}")

print("🎯 Hoàn tất! Ảnh đã xử lý được lưu trong thư mục 'processed_dataMinh'.")
