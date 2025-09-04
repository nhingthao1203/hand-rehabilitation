# hand-rehabilitation
# 🤖 Hand Gesture Recognition for Hand Rehabilitation Exercises Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-ff69b4.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project implements a **deep learning-based hand gesture recognition system** for **hand rehabilitation exercises**.  
It helps patients perform rehabilitation exercises at home by recognizing **hand movements** in real time and providing **feedback** using **computer vision** and **deep learning**.

The system leverages **CNNs** and **Transformers** to achieve **state-of-the-art accuracy** while providing a **Streamlit-based interactive demo**.

---

## 🚀 Key Features

- 🖐️ **Hand Gesture Recognition** → Detects hand gestures from webcam images/videos.
- 🎯 **Supports Multiple Models** → VGG16, ResNet50, DenseNet121, EfficientNet-B2, InceptionV3, ViT, Swin Transformer.
- ⚡ **High Accuracy** → Achieves **97.8%** accuracy using **Swin Transformer**.
- 🎥 **Real-time Recognition** → Uses **OpenCV + Mediapipe** for live gesture tracking.
- 📊 **Performance Evaluation** → Includes metrics like Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
- 🌐 **Interactive Demo** → Streamlit app for testing gestures.

---

## 📂 Project Structure

```bash
hand-gesture-rehab/
│── data/                     # Collected dataset
│── models/                   # Trained model weights
│── notebooks/                # Jupyter notebooks for experiments
│── src/                      # Source code for training & inference
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│── demo/                     # Streamlit demo app
│── requirements.txt          # Project dependencies
│── README.md                # Project documentation
## 🧠 Dataset

- **Participants:** 11 contributors  
- **Total Images:** ~12,000+ labeled samples  
- **Exercises:** 4 rehabilitation exercises → gestures for **finger flexion**, **extension**, **stretching**, and **gripping**  
- **Classes:** Multiple labeled hand positions  
- **Format:** `.jpg` images organized by gesture  
- **Data Split:** **70% Train / 15% Validation / 15% Test**

---

## 🔬 Models & Performance

| Model                | Parameters | Accuracy (%) | F1-Score | Inference Speed |
|----------------------|-----------|--------------|----------|-----------------|
| **VGG16**           | 138M      | 93.5         | 92.8     | Medium          |
| **ResNet50**        | 25M       | 95.1         | 94.7     | Fast            |
| **EfficientNet-B2** | 9M        | 96.3         | 95.9     | Fast            |
| **Vision Transformer** | 86M    | 97.2         | 96.8     | Medium          |
| **Swin Transformer** | 88M      | **97.8**     | **97.3** | Medium          |

> 🏆 **Swin Transformer** is the **best-performing model** and is used as the default for inference.

---

## 📊 Results

- **Test Accuracy:** 97.8%  
- **Real-time Inference:** < 50ms per frame  
- **Robustness:** Handles variations in **lighting, backgrounds, and skin tones**

**Example Confusion Matrix:**  
*(Insert image link here if available)*

---

## ⚡ Demo

Run the **real-time gesture recognition demo** using **Streamlit**:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/hand-gesture-rehab.git
cd hand-gesture-rehab

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run demo/app.py
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛠️ Technologies Used

- **Programming:** Python 3.9+
- **Deep Learning:** TensorFlow, PyTorch
- **Computer Vision:** OpenCV, Mediapipe
- **Visualization:** Matplotlib, Seaborn
- **Web Interface:** Streamlit
- **Training Platform:** Kaggle

---

## 📌 Future Work

- 🔹 Expand dataset with **more diverse gestures**
- 🔹 Optimize Transformer models for **faster real-time inference**
- 🔹 Develop a **cloud-based rehabilitation monitoring platform**
- 🔹 Integrate **video-based continuous gesture tracking**

---

## 👩‍💻 Authors

- **Nguyễn Thảo Nhi** *(Project Lead)*
- **Nguyễn Minh Như** *(Team Member)*  
**Supervisor:** ThS. Nguyễn Thị Mai Trang

---

## 📜 License

This project is released under the **MIT License**.  
You are free to **use, modify, and distribute** this project for **research and academic purposes**.

---

## ⭐ Acknowledgments

Special thanks to **Ho Chi Minh City Open University** and the  
**Department of Special Training** for their continuous support and guidance throughout this project.
