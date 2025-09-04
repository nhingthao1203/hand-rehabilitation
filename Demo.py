import streamlit as st
import torch
import timm
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms
import os

# ===========================
# 1️⃣ Initialize Mediapipe Hands
# ===========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# ===========================
# 2️⃣ Load EfficientNet-B2 Model
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 List of class labels (8 classes)
class_labels = [
    "1f_hand", "2f_hand", "3f_hand", "4f_hand",
    "close_hand", "duck_hand", "open_hand", "spread_hand"
]

# 🔹 Load model
model_path = "efficientnet_b2_epoch_50.pth"
num_classes = 8
model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()

# ===========================
# 3️⃣ Image Preprocessing
# ===========================
offset = 20
crop_size = (224, 224)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def enhance_contrast(img):
    """Enhance image contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)


def preprocess_image(img):
    """Preprocess image: contrast enhancement, hand detection, cropping & resizing."""
    img = enhance_contrast(img)
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
# 4️⃣ Model Prediction
# ===========================
def predict(img):
    """Run EfficientNet-B2 model to predict hand gesture."""
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output).item()

    return class_labels[predicted_class]  # Return label instead of index


# ===========================
# 5️⃣ Exercise Selection & Training Loop
# ===========================
st.title("🤖 Hand Gesture Training")
st.markdown("**🎯 Select an exercise and number of repetitions. Follow the sequence to complete the exercise!**")

# Button to switch to the Contribution Page
if st.button("📸 Contribute New Hand Gestures"):
    os.system("streamlit run contribute.py")

exercise_options = {
    "Exercise 1: ": ["close_hand", "open_hand"],
    "Exercise 2: ": ["duck_hand", "spread_hand"],
    "Exercise 3: ": ["open_hand", "spread_hand"],
    "Exercise 4: ": ["1f_hand", "2f_hand", "3f_hand", "4f_hand", "spread_hand"]
}

exercise_selected = st.selectbox("📌 Choose an exercise", list(exercise_options.keys()))
repetitions = st.number_input("🔄 Number of repetitions (1-20)", min_value=1, max_value=20, value=3, step=1)

start_button = st.button("🟢 Start Training")

# Load pose images for reference
image_folder = "images"  # Folder containing hand pose images
exercise_sequence = exercise_options[exercise_selected]
pose_images = [os.path.join(image_folder, f"{pose}.jpg") for pose in exercise_sequence]

if start_button:
    cap = cv2.VideoCapture(0)
    st.session_state.current_step = 0  # Track sequence step
    st.session_state.completed_reps = 0  # Count completed reps
    st.session_state.training_active = True  # Track training status

    # Create side-by-side layout: Webcam | Pose Guide
    col1, col2 = st.columns(2)
    with col1:
        stframe = st.empty()  # Webcam Feed
    with col2:
        pose_step = st.empty()  # Pose Guide Image

    while cap.isOpened() and st.session_state.completed_reps < repetitions:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Unable to access webcam!")
            break

        processed_img, landmarks = preprocess_image(frame)

        if processed_img is not None:
            predicted_label = predict(processed_img)

            cv2.putText(frame, f"{predicted_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            expected_pose = exercise_sequence[st.session_state.current_step]

            if predicted_label == expected_pose:
                st.session_state.current_step += 1

                if st.session_state.current_step == len(exercise_sequence):
                    st.session_state.current_step = 0
                    st.session_state.completed_reps += 1
                    st.success(f"✅ Repetition {st.session_state.completed_reps} Completed!")

            for hand_landmarks in landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display webcam feed
        with col1:
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Display pose image
        if st.session_state.current_step < len(pose_images):
            pose_img_path = pose_images[st.session_state.current_step]
            with col2:
                if os.path.exists(pose_img_path):
                    pose_step.image(pose_img_path, caption=f"Current Pose: {exercise_sequence[st.session_state.current_step]}", use_container_width=True)
                else:
                    pose_step.warning(f"⚠ Image not found: {pose_img_path}")

    cap.release()
    st.session_state.training_active = False  # Mark training as completed
    st.success("🎉 Training Completed!")
    stframe.empty()  # Clear webcam screen
    pose_step.empty()  # Clear pose guide
# Display pose images next to webcam feed
st.sidebar.header("📷 Pose Guide")
for pose_image in pose_images:
    if os.path.exists(pose_image):
        st.sidebar.image(pose_image, caption=os.path.basename(pose_image).split(".")[0], use_container_width=True)
    else:
        st.sidebar.warning(f"⚠ Image not found: {pose_image}")