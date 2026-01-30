# Smart-Border-Surveillance-AI
Human vs Animal Intrusion Detection using Computer Vision and Deep Learning

# 🛡️ Smart Border Surveillance AI

A real-world inspired **defense surveillance system** that detects motion from video feeds and classifies intrusions as **HUMAN or ANIMAL**, with support for **night / thermal vision simulation** and a real-time dashboard.

---

## 🚀 Project Overview

Border surveillance systems must work efficiently, especially during night-time operations.  
This project simulates a **smart surveillance pipeline** used in defense applications by combining:

- Motion detection (to reduce unnecessary computation)
- Deep learning–based object detection
- Human vs Animal classification
- Thermal / night vision simulation
- Real-time operator dashboard

---

## 🔍 Key Features

- ✅ Motion-triggered detection pipeline  
- 👤 Human vs 🐕 Animal intrusion classification  
- 🌙 Night / thermal vision simulation  
- 📊 Real-time Streamlit dashboard  
- ⚡ Efficient and explainable system design  

---

## 🧠 System Logic (High-Level)

1. **Motion Detection**  
   Classical computer vision is used to detect movement in video frames.

2. **Selective AI Inference**  
   Deep learning is applied only when motion is detected, reducing compute load.

3. **Object Detection & Classification**  
   YOLO-based detector identifies objects and classifies them as:
   - HUMAN
   - ANIMAL

4. **Thermal Simulation (Night Mode)**  
   Heat-map visualization simulates thermal camera output.

5. **Operator Dashboard**  
   Results are displayed in real time using Streamlit.

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- YOLO (Ultralytics)  
- Streamlit  
- NumPy  

---

## ▶️ How to Run
 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Smart-Border-Surveillance-AI.git
cd Smart-Border-Surveillance-AI
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the dashboard
streamlit run app.py
4️⃣ Upload a surveillance video
Format: MP4 / AVI

Static camera preferred

