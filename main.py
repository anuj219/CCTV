# Main file that calls other files and executes the main work
# pulls live RTSP frames, runs YOLO + zone/approach filtering, queries faces.pkl via face_recognition, logs entries to SQLite, and fires Telegram notifications.

import cv2
import numpy as np
import face_recognition
from cctv.db import init_db, should_log_encounter, log_encounter
from cctv.rtsp_stream import RTSPStream
from cctv.notifier import send_telegram_alert
from Neytra.Face.backend.ai.recognizer import load_database
from ultralytics import YOLO

import os
from dotenv import load_dotenv
load_dotenv()


init_db()
known_faces_db = load_database() # dict of {name: [128D encodings]}
yolo = YOLO("models/yolo11n.pt")


# Example 4-point polygon mapped to doorstep (calibration step)
ZONE_POINTS = np.array([[100, 200], [500, 200], [500, 600], [100, 600]], np.int32)

def is_in_zone(bbox, zone_poly):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, y2 # Bottom-center of bounding box
    return cv2.pointPolygonTest(zone_poly, (float(cx), float(cy)), False) >= 0


username = os.getenv("username")
password = os.getenv("password")
RTSP_URL = f"rtsp://${username}:${password}@192.168.29.62:554/Streaming/Channels/301"
stream = RTSPStream(RTSP_URL)

while True:
    frame = stream.read()
    if frame is None:
        continue

    # 1. YOLO Person Detection
    results = yolo(frame, classes=[0], verbose=False)
    person_boxes = []

    if results and results[0].boxes:
        for box in results[0].boxes:
            if float(box.conf[0]) >= 0.45:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if is_in_zone((x1, y1, x2, y2), ZONE_POINTS):
                    person_boxes.append((x1, y1, x2, y2))

    # 2. Extract Head Crops & Match Encodings
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for (x1, y1, x2, y2) in person_boxes:
        h = y2 - y1
        head_box = (max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y1 + int(h * 0.35)), max(0, x1))
        
        encodings = face_recognition.face_encodings(image_rgb, [head_box])
        if not encodings:
            continue

        current_enc = encodings[0]
        matched_name = "Unknown"
        best_dist = 0.6 # Distance threshold (lower = stricter)

        for name, db_encs in known_faces_db.items():
            distances = face_recognition.face_distance(db_encs, current_enc)
            min_dist = min(distances) if len(distances) > 0 else 1.0
            if min_dist < best_dist:
                best_dist = min_dist
                matched_name = name

        # 3. Log & Alert Execution
        if matched_name != "Unknown":
            if should_log_encounter(matched_name, cooldown_seconds=60):
                log_encounter(matched_name, confidence=1-best_dist, headcount=len(person_boxes))
                send_telegram_alert(f"🟢 <b>{matched_name}</b> arrived home.")
        else:
            if should_log_encounter("Unknown", cooldown_seconds=120):
                log_encounter("Unknown", confidence=0.0, headcount=len(person_boxes))
                _, img_encoded = cv2.imencode('.jpg', frame)
                send_telegram_alert("⚠️ <b>Unknown Person Detected</b> at doorstep!", img_encoded.tobytes())