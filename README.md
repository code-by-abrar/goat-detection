# 🐐 Goat Detection & Tracking System

An AI-powered computer vision system designed to detect and uniquely track goats using **YOLO**, **ByteTrack**, and **Supervision**. It features moving camera support and an on-screen counter that logs unique IDs in real-time.

---

## 📺 Project Results & Demo
Check out the project execution and video demo here:
- **[👉 View Results & Video Demo on LinkedIn](https://lnkd.in/p/dMU-mqAX)**
- Alternatively, check the [`results/` folder](file:///results/README.md).

---

## 🛠️ Tech Stack & Dependencies
- **Ultralytics YOLO** (Custom trained model: `model/best.pt`)
- **Supervision** (Visual annotations and detections)
- **OpenCV** (Video processing and UI overlay)
- **ByteTrack** (Multi-object tracking)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd Goat_detection
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the Script
Make sure your input video (e.g. `video.mp4`) and model (`model/best.pt`) are in place, then run:
```bash
python src/gem1.py
```
Press `q` to terminate the video stream and view the total unique count.
