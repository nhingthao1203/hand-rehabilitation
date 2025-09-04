import streamlit as st
import torch
import timm
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms
import os
import time
import pyttsx3
import pandas as pd
import csv
import matplotlib.pyplot as plt
import math

# ===========================
# 1️⃣ Khởi tạo Mediapipe
# ===========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# ===========================
# 2️⃣ Load mô hình EfficientNet-B2
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ["1f_hand", "2f_hand", "3f_hand", "4f_hand", "close_hand", "duck_hand", "open_hand", "spread_hand"]

model_path = "efficientnet_b2_epoch_50.pth"
num_classes = 8
model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# ===========================
# 3️⃣ Tiền xử lý ảnh
# ===========================
offset = 20
crop_size = (224, 224)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    h, w, _ = img.shape
    x_min = min([int(lm.x * w) for lm in hand_landmarks.landmark])
    y_min = min([int(lm.y * h) for lm in hand_landmarks.landmark])
    x_max = max([int(lm.x * w) for lm in hand_landmarks.landmark])
    y_max = max([int(lm.y * h) for lm in hand_landmarks.landmark])

    x1, y1 = max(0, x_min - offset), max(0, y_min - offset)
    x2, y2 = min(w, x_max + offset), min(h, y_max + offset)
    img_crop = img[y1:y2, x1:x2]

    if img_crop.shape[0] < 10 or img_crop.shape[1] < 10:
        return None, None

    img_crop = cv2.resize(img_crop, crop_size, interpolation=cv2.INTER_AREA)
    return img_crop, results.multi_hand_landmarks

# ===========================
# 4️⃣ Dự đoán bằng mô hình
# ===========================
def predict(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output).item()
    return class_labels[predicted_class]

# ===========================
# 5️⃣ Giao diện Streamlit
# ===========================
st.title("🤖 Hand Gesture Training with EfficientNet-B2")
st.markdown("**🎯 Chọn bài tập và số lần lặp lại. Thực hiện các động tác theo thứ tự!**")

exercise_options = {
    "Bài tập 1: close_hand → open_hand": ["close_hand", "open_hand"],
    "Bài tập 2: duck_hand → spread_hand": ["duck_hand", "spread_hand"],
    "Bài tập 3: open_hand → spread_hand": ["open_hand", "spread_hand"],
    "Bài tập 4: 1f_hand → 2f_hand → 3f_hand → 4f_hand → spread_hand": ["1f_hand", "2f_hand", "3f_hand", "4f_hand", "spread_hand"]
}

exercise_selected = st.selectbox("📌 Chọn bài tập", list(exercise_options.keys()))
repetitions = st.number_input("🔄 Số lần lặp lại (1-20)", min_value=1, max_value=20, value=3, step=1)

start_button = st.button("🟢 Bắt đầu tập luyện")

# Load pose images for reference
image_folder = "images"  # Folder containing hand pose images
exercise_sequence = exercise_options[exercise_selected]
pose_images = [os.path.join(image_folder, f"{pose}.jpg") for pose in exercise_sequence]

# ===========================
# 6️⃣ Bộ đếm thời gian cho mỗi bước
# ===========================
if "timer" not in st.session_state:
    st.session_state.timer = time.time()

max_time_per_step = 5  # Giới hạn thời gian mỗi bước

# ===========================
# 7️⃣ Thêm bảng xếp hạng
# ===========================
leaderboard_file = "leaderboard.csv"

# ===========================
# 8️⃣ Kích hoạt nhận diện và theo dõi tiến độ
# ===========================
if start_button:
    cap = cv2.VideoCapture(0)
    st.session_state.current_step = 0
    st.session_state.completed_reps = 0
    st.session_state.training_active = True

    col1, col2 = st.columns(2)
    with col1:
        stframe = st.empty()
    with col2:
        pose_step = st.empty()

    while cap.isOpened() and st.session_state.completed_reps < repetitions:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Không thể truy cập webcam!")
            break

        processed_img, landmarks = preprocess_image(frame)

        if processed_img is not None:
            predicted_label = predict(processed_img)

            cv2.putText(frame, f"{predicted_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            expected_pose = exercise_options[exercise_selected][st.session_state.current_step]

            if predicted_label == expected_pose:
                st.session_state.current_step += 1
                st.session_state.timer = time.time()

                if st.session_state.current_step == len(exercise_options[exercise_selected]):
                    st.session_state.current_step = 0
                    st.session_state.completed_reps += 1
                    st.success(f"✅ Lần {st.session_state.completed_reps} hoàn thành!")

            if time.time() - st.session_state.timer > max_time_per_step:
                st.warning("⏳ Hãy thực hiện động tác nhanh hơn!")

            for hand_landmarks in landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        with col1:
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if st.session_state.current_step < len(exercise_options[exercise_selected]):
            with col2:
                pose_step.text(f"Động tác tiếp theo: {exercise_options[exercise_selected][st.session_state.current_step]}")

    cap.release()
    st.session_state.training_active = False
    st.success("🎉 Tập luyện hoàn thành!")

    # Lưu thời gian tập luyện vào bảng xếp hạng
    completion_time = time.time() - st.session_state.timer
    with open(leaderboard_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Người chơi", completion_time])

    df = pd.read_csv(leaderboard_file, names=["Người chơi", "Thời gian"])
    st.table(df.sort_values(by="Thời gian", ascending=True).head(5))
st.sidebar.header("📷 Pose Guide")
for pose_image in pose_images:
    if os.path.exists(pose_image):
        st.sidebar.image(pose_image, caption=os.path.basename(pose_image).split(".")[0], use_container_width=True)
    else:
        st.sidebar.warning(f"⚠ Image not found: {pose_image}")