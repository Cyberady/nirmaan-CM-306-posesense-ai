from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import mediapipe as mp
import numpy as np
import math, io, time, json, base64
from datetime import datetime
from collections import deque
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

import os, pathlib

app = Flask(__name__)

# ── Persistent history stored in a JSON file next to app.py ──────────────────
HISTORY_FILE = pathlib.Path(__file__).parent / "session_history.json"

def load_history():
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text())
    except Exception:
        pass
    return []

def save_history(history):
    try:
        HISTORY_FILE.write_text(json.dumps(history, indent=2))
    except Exception as e:
        print(f"[PoseSense] Could not save history: {e}")

mp_pose      = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

pose      = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
    max_num_faces=1
)

# ── Session state ────────────────────────────────────────────────────────────
session_data = {
    "posture_score": 0,
    "movement_score": 0,
    "eye_contact_score": 0,
    "nervousness_level": 0,
    "confidence_score": 0,
    "presentation_score": 0,
    "emotion": "neutral",
    "emotion_emoji": "😐",
    "gesture": "natural",
    "gesture_label": "Natural",
    "feedback": [],
    "alerts": [],
    "confidence_history": [],
    "posture_history": [],
    "eye_history": [],
    "heatmap_history": [],
    "session_start": None,
    "frame_count": 0,
    "mode": "general",
    "first_30_avg": None,
    "last_30_avg": None,
    "session_notes": [],
    "goals_hit": [],
    "filler_count": 0,
    "snapshot_frame": None,
    "paused": False,
    # Biometric signals exposed to frontend
    "blink_rate": 0,
    "blink_status": "normal",
    "head_nod": "stable",
    "head_nod_label": "Stable",
}

conf_window    = deque(maxlen=300)
posture_window = deque(maxlen=300)
eye_window     = deque(maxlen=300)
move_window    = deque(maxlen=30)
heatmap_win    = deque(maxlen=7200)   # ~2hrs at 1fps
prev_lm        = None
alert_cd       = {}
camera         = None
last_frame_raw = None

# ── Blink & head-nod tracking state ─────────────────────────────────────────
blink_counter   = 0
blink_total     = 0
blink_window    = deque(maxlen=60)   # last 60 seconds of per-second counts
blink_frame_buf = deque(maxlen=5)    # EAR values for blink detection
blink_in_blink  = False
blink_ts        = time.time()        # timestamp for per-second counting

nose_y_window   = deque(maxlen=30)   # nose Y for head-nod detection

EMOTION_MAP = {
    "confident": "😎",
    "happy": "😊",
    "neutral": "😐",
    "nervous": "😰",
    "stressed": "😟",
    "focused": "🧐",
}

# ── Camera ───────────────────────────────────────────────────────────────────
def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def d2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def lpt(lm, i, w, h):
    p = lm[i]
    return (p.x * w, p.y * h)

# ── Analysis functions ────────────────────────────────────────────────────────
def analyze_posture(lm, w, h):
    try:
        ls = lpt(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
        rs = lpt(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        le = lpt(lm, mp_pose.PoseLandmark.LEFT_EAR.value, w, h)
        re = lpt(lm, mp_pose.PoseLandmark.RIGHT_EAR.value, w, h)
        sw = max(d2(ls, rs), 1)
        slope  = abs(ls[1] - rs[1]) / sw
        offset = abs((le[0] + re[0]) / 2 - (ls[0] + rs[0]) / 2) / sw
        tilt   = abs(le[1] - re[1]) / sw
        score  = 100 - min(40, slope * 200) - min(30, offset * 150) - min(20, tilt * 150)
        fb = []
        if slope > 0.10:  fb.append("Level your shoulders — they appear uneven")
        if offset > 0.15: fb.append("Pull your head back — avoid forward leaning")
        if tilt > 0.10:   fb.append("Keep your head upright and centered")
        return max(0, min(100, score)), fb
    except Exception:
        return 50, []

def analyze_movement(lm, prev):
    if prev is None:
        return 50, "natural", []
    try:
        idxs = [
            mp_pose.PoseLandmark.LEFT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        ]
        total = sum(
            math.sqrt((lm[i].x - prev[i].x) ** 2 + (lm[i].y - prev[i].y) ** 2)
            for i in idxs
        )
        move_window.append(total)
        avg = sum(move_window) / len(move_window)
        if   avg < 0.004: return 40,  "still",     ["Add natural hand gestures to look engaged"]
        elif avg < 0.018: return 95,  "natural",   []
        elif avg < 0.045: return 70,  "active",    ["Slightly reduce hand movement"]
        else:             return 25,  "excessive",  ["Reduce hand movement — signals nervousness"]
    except Exception:
        return 50, "natural", []

def analyze_eye_contact(fl, w, h):
    try:
        L = [362, 385, 387, 263, 373, 380]
        R = [33, 160, 158, 133, 153, 144]
        lep = [(fl.landmark[i].x * w, fl.landmark[i].y * h) for i in L]
        rep = [(fl.landmark[i].x * w, fl.landmark[i].y * h) for i in R]

        def ear(p):
            return (d2(p[1], p[5]) + d2(p[2], p[4])) / (2 * max(d2(p[0], p[3]), 0.001))

        avg_ear = (ear(lep) + ear(rep)) / 2
        nt = fl.landmark[4]
        lc = fl.landmark[234]
        rc = fl.landmark[454]
        fw = max(abs(rc.x - lc.x), 0.01)
        noff = abs(nt.x - (lc.x + rc.x) / 2) / fw
        score = 100 - (30 if avg_ear < 0.20 else 0) - min(40, noff * 200)
        fb = []
        if avg_ear < 0.20: fb.append("Keep your eyes open — maintain eye contact")
        if noff > 0.15:    fb.append("Look directly at the camera")
        return max(0, min(100, score)), fb, avg_ear
    except Exception:
        return 50, [], 0.25

def detect_blink(avg_ear):
    """
    Detects blinks from EAR values. Updates global blink counters.
    Returns per-minute blink rate and status string.
    """
    global blink_in_blink, blink_total, blink_ts

    BLINK_THRESH = 0.18
    blink_frame_buf.append(avg_ear)

    # Detect blink onset (EAR drops below threshold)
    if avg_ear < BLINK_THRESH and not blink_in_blink:
        blink_in_blink = True
    elif avg_ear >= BLINK_THRESH and blink_in_blink:
        blink_in_blink = False
        blink_total += 1

        # Record this blink in the per-second bucket
        now = time.time()
        elapsed = now - blink_ts
        if elapsed >= 1.0:
            blink_window.append(blink_total)
            blink_total = 0
            blink_ts = now

    # Compute blink rate over last ~60s
    if len(blink_window) > 0:
        rate = int(sum(blink_window) / max(len(blink_window), 1) * 60)
    else:
        rate = 0

    # Classify status
    if rate < 8:
        status = "low"
    elif rate > 30:
        status = "high"
    else:
        status = "normal"

    return rate, status

def detect_head_nod(fl):
    """
    Tracks nose-tip Y position over time to classify head nod pattern.
    Returns nod state and label.
    """
    try:
        nose_y = fl.landmark[4].y
        nose_y_window.append(nose_y)
        if len(nose_y_window) < 10:
            return "stable", "Stable"
        variance = float(np.var(list(nose_y_window)))
        if variance > 0.00015:
            return "nodding", "Nodding ✓"
        elif variance < 0.000005:
            return "frozen", "Frozen ⚠"
        else:
            return "stable", "Stable"
    except Exception:
        return "stable", "Stable"

def detect_emotion(fl, posture_s, eye_s, nervousness):
    """
    Robust 6-emotion classifier using normalised face landmarks.
    """
    try:
        lmc = fl.landmark[61]
        rmc = fl.landmark[291]
        mouth_w = abs(rmc.x - lmc.x)

        lc  = fl.landmark[234]
        rc  = fl.landmark[454]
        face_w = max(abs(rc.x - lc.x), 0.01)

        smile_ratio = mouth_w / face_w

        upper_lip = fl.landmark[13]
        lower_lip = fl.landmark[14]
        mouth_open = abs(lower_lip.y - upper_lip.y) / max(face_w, 0.01)

        lb   = fl.landmark[105]; rb   = fl.landmark[334]
        let_ = fl.landmark[159]; ret_ = fl.landmark[386]
        brow_raise = ((let_.y - lb.y) + (ret_.y - rb.y)) / 2

        def ear(pts):
            p = [fl.landmark[i] for i in pts]
            return (abs(p[1].y - p[5].y) + abs(p[2].y - p[4].y)) / (2 * max(abs(p[0].x - p[3].x), 0.001))

        avg_ear = (ear([362, 385, 387, 263, 373, 380]) + ear([33, 160, 158, 133, 153, 144])) / 2

        if nervousness > 65:
            return "nervous", "😰"
        if nervousness > 42 and brow_raise < 0.005:
            return "stressed", "😟"
        if smile_ratio > 0.44 and avg_ear > 0.18:
            return "happy", "😊"
        if posture_s > 72 and eye_s > 68 and nervousness < 28:
            return "confident", "😎"
        if brow_raise > 0.030 and avg_ear > 0.20:
            return "focused", "🧐"
        return "neutral", "😐"

    except Exception:
        return "neutral", "😐"

def check_alerts(ps, es, ms, nv, blink_rate, head_nod):
    now = time.time()
    C = 15
    alerts = []

    def may(key, msg, sev="warning"):
        if now - alert_cd.get(key, 0) > C:
            alert_cd[key] = now
            alerts.append({"key": key, "msg": msg, "severity": sev, "ts": int(now)})

    if ps < 55:         may("slouch",  "⚠ You've been slouching — sit up straight!")
    if es < 45:         may("gaze",    "👁 Maintain eye contact with the camera")
    if ms < 30:         may("fidget",  "✋ Reduce hand fidgeting — take a deep breath")
    if nv > 70:         may("nervous", "💙 Breathe slowly — you're appearing nervous", "info")
    if blink_rate < 8 and blink_rate > 0:
                        may("blink_low",  "👁 Blink more naturally — you look stiff", "warning")
    if blink_rate > 30: may("blink_high", "👁 Rapid blinking detected — take a breath", "info")
    if head_nod == "frozen":
                        may("frozen",  "🗿 Add natural head movement — avoid looking robotic", "warning")
    if ps > 85 and es > 80:
                        may("great",   "⭐ Excellent posture & eye contact!", "success")
    return alerts

# ── Frame generator ──────────────────────────────────────────────────────────
def generate_frames():
    global prev_lm, session_data, last_frame_raw
    cam = get_camera()
    while True:
        if session_data.get("paused", False):
            time.sleep(0.1)
            if last_frame_raw is not None:
                ret, buf = cv2.imencode('.jpg', last_frame_raw, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            continue

        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = frame.shape[:2]
        pr    = pose.process(rgb)
        fr    = face_mesh.process(rgb)

        ps = ms = es = 50
        fb   = []
        em   = "neutral"
        emoji = "😐"
        gest = "natural"
        gl   = "Natural"
        avg_ear = 0.25
        blink_rate = session_data.get("blink_rate", 0)
        blink_status = session_data.get("blink_status", "normal")
        head_nod = session_data.get("head_nod", "stable")
        head_nod_label = session_data.get("head_nod_label", "Stable")

        if pr.pose_landmarks:
            lm = pr.pose_landmarks.landmark
            ps, p_fb = analyze_posture(lm, w, h)
            ms, gest, m_fb = analyze_movement(lm, prev_lm)
            gl = {
                "natural": "Natural ✓",
                "excessive": "Too Active",
                "still": "Too Still",
                "active": "Active",
            }.get(gest, "Natural")
            fb.extend(p_fb)
            fb.extend(m_fb)
            prev_lm = lm
            mp_drawing.draw_landmarks(
                frame,
                pr.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2),
            )

        if fr.multi_face_landmarks:
            fl = fr.multi_face_landmarks[0]
            es, e_fb, avg_ear = analyze_eye_contact(fl, w, h)
            nv = max(0, min(100, (
                max(0, 60 - ps) * 0.35 +
                max(0, 60 - ms) * 0.35 +
                max(0, 60 - es) * 0.30
            )))
            em, emoji = detect_emotion(fl, ps, es, nv)
            blink_rate, blink_status = detect_blink(avg_ear)
            head_nod, head_nod_label = detect_head_nod(fl)
            fb.extend(e_fb)
        else:
            nv = 50

        conf = ps * 0.35 + ms * 0.25 + es * 0.25 + (100 - nv) * 0.15
        pres = ps * 0.30 + ms * 0.20 + es * 0.30 + conf * 0.20

        conf_window.append(round(conf))
        posture_window.append(round(ps))
        eye_window.append(round(es))
        heatmap_win.append(round(conf))

        fc = session_data["frame_count"] + 1
        if fc == 30:
            session_data["first_30_avg"] = round(sum(list(conf_window)[:30]) / 30)
        if fc > 30:
            session_data["last_30_avg"] = round(sum(list(conf_window)[-30:]) / 30)

        alerts = check_alerts(ps, es, ms, nv, blink_rate, head_nod)

        session_data.update({
            "posture_score":      round(ps),
            "movement_score":     round(ms),
            "eye_contact_score":  round(es),
            "nervousness_level":  round(nv),
            "confidence_score":   round(conf),
            "presentation_score": round(pres),
            "emotion":            em,
            "emotion_emoji":      emoji,
            "gesture":            gest,
            "gesture_label":      gl,
            "feedback":           list(dict.fromkeys(fb))[:3],
            "alerts":             alerts,
            "confidence_history": list(conf_window)[-60:],
            "posture_history":    list(posture_window)[-60:],
            "eye_history":        list(eye_window)[-60:],
            "heatmap_history":    list(heatmap_win)[-300:],
            "frame_count":        fc,
            "blink_rate":         blink_rate,
            "blink_status":       blink_status,
            "head_nod":           head_nod,
            "head_nod_label":     head_nod_label,
        })

        # HUD overlay
        cv2.rectangle(frame, (0, h - 44), (w, h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Conf:{round(conf)}%  Post:{round(ps)}%  Eye:{round(es)}%  {em}",
            (8, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1,
        )

        last_frame_raw = frame.copy()
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    mode = request.args.get('mode', 'general')
    session_data.update({
        "session_start":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "confidence_history": [],
        "posture_history":    [],
        "eye_history":        [],
        "heatmap_history":    [],
        "frame_count":        0,
        "alerts":             [],
        "first_30_avg":       None,
        "last_30_avg":        None,
        "mode":               mode,
        "session_notes":      [],
        "goals_hit":          [],
        "filler_count":       0,
        "snapshot_frame":     None,
        "paused":             False,
        "blink_rate":         0,
        "blink_status":       "normal",
        "head_nod":           "stable",
        "head_nod_label":     "Stable",
    })
    conf_window.clear()
    posture_window.clear()
    eye_window.clear()
    heatmap_win.clear()
    blink_window.clear()
    nose_y_window.clear()
    return render_template('dashboard.html', mode=mode)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )

@app.route('/metrics')
def metrics():
    return jsonify(session_data)

# ── Snapshot ─────────────────────────────────────────────────────────────────
@app.route('/snapshot')
def snapshot():
    global last_frame_raw
    if last_frame_raw is None:
        return jsonify({"error": "No frame available"}), 404

    frame = last_frame_raw.copy()
    h, w = frame.shape[:2]
    d = session_data

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (340, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, "PoseSense AI", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (167, 139, 250), 2)

    scores = [
        (f"Confidence:  {d['confidence_score']}%",  (0, 255, 180)),
        (f"Posture:     {d['posture_score']}%",      (167, 139, 250)),
        (f"Eye Contact: {d['eye_contact_score']}%",  (34, 211, 238)),
        (f"Emotion:     {d['emotion']}",             (245, 158, 11)),
    ]
    for i, (txt, clr) in enumerate(scores):
        cv2.putText(frame, txt, (20, 58 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 80, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buf).decode('utf-8')
    return jsonify({"image": "data:image/jpeg;base64," + b64, "timestamp": ts})

# ── Session notes ─────────────────────────────────────────────────────────────
@app.route('/add_note', methods=['POST'])
def add_note():
    note = request.json.get('note', '').strip()
    if note:
        session_data["session_notes"].append({
            "time":       datetime.now().strftime("%H:%M:%S"),
            "note":       note,
            "confidence": session_data["confidence_score"],
        })
    return jsonify({"ok": True, "notes": session_data["session_notes"]})

@app.route('/get_notes')
def get_notes():
    return jsonify(session_data["session_notes"])

# ── Goal achieved ─────────────────────────────────────────────────────────────
@app.route('/goal_hit', methods=['POST'])
def goal_hit():
    g = request.json.get('goal', '')
    if g not in session_data["goals_hit"]:
        session_data["goals_hit"].append(g)
    return jsonify({"ok": True})

# ── Filler word count ─────────────────────────────────────────────────────────
@app.route('/update_fillers', methods=['POST'])
def update_fillers():
    session_data["filler_count"] = request.json.get('count', 0)
    return jsonify({"ok": True})

# ── Pause toggle ──────────────────────────────────────────────────────────────
@app.route('/pause', methods=['POST'])
def pause():
    session_data['paused'] = not session_data.get('paused', False)
    return jsonify({"paused": session_data['paused']})

# ── PDF Report ────────────────────────────────────────────────────────────────
@app.route('/generate_report', methods=['POST'])
def generate_report():
    d = request.json or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        rightMargin=60, leftMargin=60, topMargin=60, bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    PURPLE = colors.HexColor('#7C3AED')
    PL     = colors.HexColor('#A78BFA')
    DARK   = colors.HexColor('#1E1B4B')
    GRAY   = colors.HexColor('#6B7280')

    def sty(n, **kw):
        return ParagraphStyle(n, parent=styles['Normal'], **kw)

    story = [
        Paragraph(
            "PoseSense AI",
            sty('T', fontSize=26, textColor=PURPLE, spaceAfter=4, fontName='Helvetica-Bold'),
        ),
        Paragraph(
            f"Interview Performance Report · {d.get('timestamp', '')}",
            sty('S', fontSize=12, textColor=GRAY, spaceAfter=16),
        ),
        HRFlowable(width="100%", thickness=1, color=PL, spaceAfter=12),
        Paragraph(
            "Performance Metrics",
            sty('H2', fontSize=15, textColor=DARK, spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold'),
        ),
    ]

    rows = [["Metric", "Score", "Rating"]]
    for name, key in [
        ("Posture",       "posture"),
        ("Movement",      "movement"),
        ("Eye Contact",   "eye_contact"),
        ("Confidence",    "confidence"),
        ("Presentation",  "presentation"),
        ("Nervousness",   "nervousness"),
    ]:
        v = d.get(key, 0)
        rows.append([name, f"{v}%", _rate(name, v)])

    t = Table(rows, colWidths=[2.8 * inch, 1.4 * inch, 2.2 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, 0), 11),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F5F3FF'), colors.white]),
        ('GRID',         (0, 0), (-1, -1), 0.4, colors.HexColor('#DDD6FE')),
        ('ALIGN',        (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING',      (0, 0), (-1, -1), 7),
    ]))
    story += [t, Spacer(1, 0.2 * inch)]

    # Behavioural signals
    story += [
        Paragraph(
            "Behavioural Signals",
            sty('H2', fontSize=15, textColor=DARK, spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold'),
        ),
        Paragraph(
            f"Detected emotion: <b>{d.get('emotion', 'neutral').capitalize()}</b> {d.get('emotion_emoji', '😐')}",
            sty('B', fontSize=11, spaceAfter=5),
        ),
        Paragraph(
            f"Gesture pattern: <b>{d.get('gesture_label', 'Natural')}</b>",
            sty('B', fontSize=11, spaceAfter=5),
        ),
    ]

    if d.get('blink_rate'):
        story.append(Paragraph(
            f"Blink rate: <b>{d['blink_rate']}/min</b> ({d.get('blink_status', 'normal')})",
            sty('B', fontSize=11, spaceAfter=5),
        ))
    if d.get('head_nod_label'):
        story.append(Paragraph(
            f"Head movement: <b>{d['head_nod_label']}</b>",
            sty('B', fontSize=11, spaceAfter=5),
        ))
    if d.get('filler_count', 0) > 0:
        story.append(Paragraph(
            f"Filler words detected: <b>{d['filler_count']}</b> (um / uh / like / you know)",
            sty('B', fontSize=11, spaceAfter=5),
        ))
    if d.get('first_30_avg') and d.get('last_30_avg'):
        delta = d['last_30_avg'] - d['first_30_avg']
        arr = "↑ improved" if delta > 0 else ("↓ declined" if delta < 0 else "→ stable")
        story.append(Paragraph(
            f"Confidence trend: <b>{d['first_30_avg']}% → {d['last_30_avg']}%</b> ({arr})",
            sty('HL', fontSize=11, textColor=PURPLE, spaceAfter=5),
        ))

    # AI Feedback
    story += [
        Spacer(1, 0.1 * inch),
        Paragraph(
            "AI Feedback",
            sty('H2', fontSize=15, textColor=DARK, spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold'),
        ),
    ]
    for fb_item in (d.get('feedback') or ["Great session! Keep it up."]):
        story.append(Paragraph(f"• {fb_item}", sty('B', fontSize=11, spaceAfter=5)))

    # Session notes
    notes = d.get('session_notes', [])
    if notes:
        story += [
            Spacer(1, 0.1 * inch),
            Paragraph(
                "Session Notes",
                sty('H2', fontSize=15, textColor=DARK, spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold'),
            ),
        ]
        for n in notes:
            story.append(Paragraph(
                f"[{n['time']}] {n['note']} <i>(Confidence: {n['confidence']}%)</i>",
                sty('B', fontSize=10, spaceAfter=4),
            ))

    # Overall assessment
    conf = d.get('confidence', 0)
    if   conf >= 80: verdict = "Outstanding. You projected strong confidence throughout."
    elif conf >= 65: verdict = "Solid performance. A few tweaks will elevate your interviews significantly."
    elif conf >= 50: verdict = "Developing. Focus on the feedback and practice daily with PoseSense AI."
    else:            verdict = "Needs work. Regular sessions will build your body-language skills."

    story += [
        Spacer(1, 0.1 * inch),
        Paragraph(
            "Overall Assessment",
            sty('H2', fontSize=15, textColor=DARK, spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold'),
        ),
        Paragraph(verdict, sty('B', fontSize=11, spaceAfter=5)),
        Spacer(1, 0.3 * inch),
        HRFlowable(width="100%", thickness=0.5, color=PL),
        Paragraph("Generated by PoseSense AI", sty('F', fontSize=9, textColor=GRAY)),
    ]

    doc.build(story)
    buf.seek(0)
    return send_file(
        buf, mimetype='application/pdf',
        as_attachment=True, download_name='posesense_report.pdf',
    )

def _rate(n, v):
    if "Nervousness" in n:
        return "Very Calm ✓" if v <= 20 else "Calm" if v <= 40 else "Moderate" if v <= 60 else "High"
    return "Excellent ✓" if v >= 80 else "Good" if v >= 65 else "Fair" if v >= 50 else "Needs Work"

def _grade(conf):
    if conf >= 85: return "A+"
    if conf >= 78: return "A"
    if conf >= 70: return "B+"
    if conf >= 62: return "B"
    if conf >= 55: return "C+"
    if conf >= 48: return "C"
    return "D"

# ── Save session ──────────────────────────────────────────────────────────────
@app.route('/save_session', methods=['POST'])
def save_session():
    d = request.json or {}
    history = load_history()
    entry = {
        "date":          datetime.now().strftime("%b %d, %Y %H:%M"),
        "mode":          d.get("mode", "general"),
        "confidence":    d.get("confidence", 0),
        "posture":       d.get("posture", 0),
        "eye_contact":   d.get("eye_contact", 0),
        "nervousness":   d.get("nervousness", 0),
        "duration_s":    d.get("duration_s", 0),
        "emotion":       d.get("emotion", "neutral"),
        "emotion_emoji": d.get("emotion_emoji", "😐"),
        "filler_count":  d.get("filler_count", 0),
        "grade":         _grade(d.get("confidence", 0)),
    }
    history.append(entry)
    history = history[-20:]
    save_history(history)
    return jsonify({"ok": True, "total": len(history)})

@app.route('/history')
def history():
    return jsonify(load_history())

@app.route('/clear_history', methods=['POST'])
def clear_history():
    save_history([])
    return jsonify({"ok": True})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)