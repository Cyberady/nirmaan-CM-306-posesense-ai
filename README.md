# PoseSense AI 🎯

**AI-Powered Interview Performance Analyzer**

A real-time body language analysis platform that helps you ace interviews through live pose estimation, eye contact tracking, and AI coaching.

---

## Features

- **Real-Time Pose Analysis** — MediaPipe tracks 33 body landmarks for posture scoring
- **Eye Contact Detection** — Face mesh tracks gaze and blink rate
- **Movement Analysis** — Detects excessive hand gestures vs. natural movement
- **Nervousness Score** — Composite metric from posture + movement + eye contact
- **Confidence Trend Graph** — Chart.js visualization of confidence over time
- **AI Interview Coach** — GPT-powered chat for mock questions and tips
- **PDF Reports** — Downloadable session performance summary

---

## Setup & Run

### Requirements
- Python 3.8+
- Webcam/camera
- Internet connection (for AI Coach)

### Install & Start
```bash
chmod +x start.sh
./start.sh
```

Or manually:
```bash
pip install flask opencv-python mediapipe reportlab
python app.py
```

### Access
- **Landing Page**: http://localhost:5000/
- **Dashboard**: http://localhost:5000/dashboard

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python Flask |
| Pose Analysis | MediaPipe + OpenCV |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Charts | Chart.js 4 |
| AI Coach | GPT via PicoApps API |
| PDF Reports | ReportLab |
| Typography | Syne + DM Sans + JetBrains Mono |

---

## Metrics Explained

| Metric | How It's Calculated |
|--------|-------------------|
| **Posture** | Shoulder slope + head forward offset + head tilt |
| **Movement** | Wrist/elbow velocity averaged over 30 frames |
| **Eye Contact** | Eye aspect ratio + nose position offset |
| **Nervousness** | Weighted composite of posture, movement, eye contact |
| **Confidence** | 35% posture + 25% movement + 25% eye contact + 15% (100-nervousness) |
| **Presentation** | 30% posture + 20% movement + 30% eye contact + 20% confidence |

---

## API Endpoints

- `GET /` — Landing page
- `GET /dashboard` — Main dashboard
- `GET /video_feed` — MJPEG webcam stream with overlay
- `GET /metrics` — JSON metrics (polled every second)
- `POST /generate_report` — Generate PDF report