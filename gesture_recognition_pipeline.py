import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)
dir_str = ['open_hand', 'close_hand', 'duck_hand', 'spread_hand',
           '1f_hand', '2f_hand', '3f_hand', '4f_hand']
index = 0
name_hand_pose = dir_str[index]
name_user = 'dataMain'
base_dir = rf"D:/gesture_recognition/{name_user}"
save_dir = os.path.join(base_dir, name_hand_pose)
os.makedirs(save_dir, exist_ok=True)

# Kiểm tra nếu tất cả các thư mục có >= 200 ảnh thì thoát chương trình
if all(len(os.listdir(os.path.join(base_dir, d))) >= 200 for d in dir_str):
    print("Tất cả thư mục đều đầy đủ 200 ảnh. Dừng chương trình.")
    exit()

saving = False
last_save_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không thể lấy khung hình từ camera!")
        break

    count = len(os.listdir(save_dir))

    # Kiểm tra lại nếu tất cả thư mục đã đầy đủ 200 ảnh
    if all(len(os.listdir(os.path.join(base_dir, d))) >= 500 for d in dir_str):
        print("Tất cả thư mục đều đầy đủ 200 ảnh. Đang thoát...")
        break

    # Chuyển ảnh sang RGB để Mediapipe xử lý
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_img = None
    flag = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            w_hand = x_max - x_min
            h_hand = y_max - y_min
            area_hand = w_hand * h_hand

            x_min = max(x_min - 50, 0)
            y_min = max(y_min - 50, 0)
            x_max = min(x_max + 50, w)
            y_max = min(y_max + 50, h)

            if w_hand > 224 or h_hand > 224 or w_hand < 60 or h_hand < 60 or area_hand < 96 * 96:
                flag = False
                msg = "Move your hand further!" if (w_hand > 224 or h_hand > 224) else "Move your hand closer!"
                cv2.putText(frame, f"Error: {msg}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                hand_img = frame[y_min:y_max, x_min:x_max]

    text_folder = f"Saving to: {name_hand_pose}/"
    text_count = f"Image #: {count}"
    text_mode = "Saving: ON" if saving else "Saving: OFF"

    cv2.putText(frame, text_folder, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, text_count, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, text_mode, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if saving else (255, 0, 0), 2)

    cv2.imshow("Original", frame)

    if hand_img is not None and hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
        cv2.imshow("Hand Cropped", hand_img)

    if saving and flag and hand_img is not None and time.time() - last_save_time > 0.05:
        img_name = os.path.join(save_dir, f"hand_{count}.jpg")
        cv2.imwrite(img_name, hand_img)
        print(f"Đã lưu ảnh: {img_name}")
        last_save_time = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        saving = True
        print("Bắt đầu lưu ảnh liên tục...")

    elif key == ord('p') or len(os.listdir(save_dir)) >= 500:
        saving = False
        index = (index + 1) % len(dir_str)
        name_hand_pose = dir_str[index]
        save_dir = os.path.join(base_dir, name_hand_pose)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Đã chuyển sang thư mục: {save_dir}")

    elif key == ord('q'):
        break

    elif key == ord('c'):
        index = (index + 1) % len(dir_str)
        name_hand_pose = dir_str[index]
        save_dir = os.path.join(base_dir, name_hand_pose)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Đã chuyển sang thư mục: {save_dir}")

cap.release()
cv2.destroyAllWindows()
