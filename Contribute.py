import streamlit as st
import cv2
import os
import mediapipe as mp
import numpy as np
import time
from PIL import Image
import uuid  # Thêm dòng này


# ===========================
# 1️⃣ Initialize Mediapipe Hands
# ===========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# ===========================
# 2️⃣ Define Contribution Directory
# ===========================
contribution_dir = "contribution"
os.makedirs(contribution_dir, exist_ok=True)

# ===========================
# 3️⃣ UI for Contribution Page
# ===========================
st.title("🙌 Contribute Hand Gesture Data")
st.markdown("📸 **Capture images to contribute new hand gestures.**")

if st.button("📸 Back to Trainning Page", key="button_2"):
    os.system("streamlit run Demo.py")
# 🔹 Existing predefined classes
predefined_classes = [
    "1f_hand", "2f_hand", "3f_hand", "4f_hand",
    "close_hand", "duck_hand", "open_hand", "spread_hand"
]

# 🔹 Get user-contributed classes from the "contribution" directory
user_contributed_classes = sorted(
    [d for d in os.listdir(contribution_dir) if os.path.isdir(os.path.join(contribution_dir, d))]
)

# 🔹 Combine both predefined & user-contributed classes
all_classes = predefined_classes + user_contributed_classes

# 🔹 User selects an existing class or creates a new one
selected_class = st.selectbox("📂 Choose a gesture class:", all_classes + ["Create New Class"])

if selected_class == "Create New Class":
    new_class = st.text_input("✍ Enter new class name:")
    if new_class and new_class not in all_classes:
        selected_class = new_class
        os.makedirs(os.path.join(contribution_dir, selected_class), exist_ok=True)
        st.success(f"✅ New class {selected_class} created!")
    elif new_class in all_classes:
        st.warning("⚠ Class name already exists! Please choose a different name.")
    else:
        st.warning("⚠ Please enter a class name to create.")

# ===========================
# 4️⃣ Capture and Save Images
# ===========================
# Khởi tạo trạng thái nếu chưa có
if "button_state" not in st.session_state:
    st.session_state.button_state = False

# Gán giá trị cho button tránh gọi nhiều lần
start_capture = st.button("📷 Start Capturing Images", key="button_3")


if start_capture and selected_class and selected_class != "Create New Class":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    count = 0
    last_no_hand_warning = time.time()  # Track last "No Hand Detected" warning

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Unable to access webcam!")
            break

        # Convert BGR to RGB for Mediapipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(img_rgb)
        hand_detected = results.multi_hand_landmarks is not None

        # Draw hand landmarks if detected
        if hand_detected:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show "No Hand Detected" warning only once every 2 seconds
        else:
            cv2.putText(frame, f"No hand detected! Make sure your hand is visible", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


        # Display the frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)


        # Save image if a hand is detected
        if hand_detected:
            save_path = os.path.join(contribution_dir, selected_class, f"{selected_class}_{count}.jpg")
            cv2.imwrite(save_path, frame)
            count += 1
            st.success(f"✅ Image {count}/10 saved: {save_path}")

        if count >= 10:  # Limit to 10 images per session
            st.success(f"🎉 10 images saved for class {selected_class}. Thank you for contributing!")
            st.session_state.button_state = not st.session_state.button_state
            break

    cap.release()
    stframe.empty()  # Clear webcam feed