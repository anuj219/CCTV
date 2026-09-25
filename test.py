"""Local CCTV test runner.

Shows the live stream with person bounding boxes and prints recognition results.
This file intentionally does not send Telegram notifications or write encounter logs.

Run from the project root:
    python -m cctv.test

Optional environment variables:
    RTSP_URL       Full camera URL. If omitted, builds one from username/password.
    username      Camera username.
    password      Camera password.
    CAMERA_URL    Set to a local video file or camera index, such as 0.

Press q or Escape in the video window to stop.
"""

import time

import os
from collections import Counter, deque

import cv2
import face_recognition
import numpy as np
from dotenv import load_dotenv
from Neytra.Face.backend.ai.recognizer import load_database
from ultralytics import YOLO

from cctv.rtsp_stream import RTSPStream

load_dotenv()

PERSON_CONFIDENCE = 0.45
MATCH_THRESHOLD = 0.42
MIN_MATCH_MARGIN = 0.015
MATCH_SAMPLE_COUNT = 3
VOTE_WINDOW_SIZE = 3
MINIMUM_VOTES = 2
RECOGNITION_INTERVAL = 3
TRACK_MAX_MISSED_FRAMES = 30
MODEL_PATH = os.path.join("models", "yolo11n.pt")

# Replace these points with the calibrated doorstep polygon.
# landscape mode ke liye -
# ZONE_POINTS = np.array([[1000, 500], [1200, 150], [2000, 500], [1450, 1100]], np.int32)
# portrait ke liye - 
ZONE_POINTS = np.array([[100, 1000], [800, 700],  [800, 2100], [100, 2100]], np.int32)


def is_in_zone(bbox, zone_poly):
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) // 2, y2)
    return cv2.pointPolygonTest(zone_poly, center, False) >= 0


def get_stream_source():
    source = os.getenv("CAMERA_URL")
    if source:
        try:
            return int(source)
        except ValueError:
            return source

    configured_url = os.getenv("RTSP_URL")
    if configured_url:
        return configured_url

    username = os.getenv("uname", "")
    password = os.getenv("password", "")
    # print("[TEST] url : "+ f"rtsp://{username}:{password}@192.168.29.62:554/Streaming/Channels/301")
    return f"rtsp://{username}:{password}@192.168.29.62:554/Streaming/Channels/301"


def match_face(image_rgb, person_box, known_faces_db):
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    image_height, image_width = image_rgb.shape[:2]
    crop_x1 = max(0, x1)
    crop_x2 = min(image_width, x2)
    crop_y1 = max(0, y1)
    crop_y2 = min(image_height, y1 + int(height * 0.5))
    head_crop = np.ascontiguousarray(
        image_rgb[crop_y1:crop_y2, crop_x1:crop_x2],
        dtype=np.uint8,
    )

    encodings = []
    if head_crop.size:
        face_locations = face_recognition.face_locations(head_crop, model="hog")
        if face_locations:
            largest_face = max(
                face_locations,
                key=lambda location: (location[2] - location[0]) * (location[1] - location[3]),
            )
            encodings = face_recognition.face_encodings(head_crop, [largest_face])

    # Tiny or blurred CCTV faces may not be detected by HOG. Preserve a
    # fallback for the same upper-body crop used by the enrollment script.
    if not encodings:
        fallback_box = (
            crop_y1,
            crop_x2,
            min(image_height, y1 + int(height * 0.35)),
            crop_x1,
        )
        encodings = face_recognition.face_encodings(image_rgb, [fallback_box])
    if not encodings:
        return "No face", None

    current_encoding = encodings[0]
    candidates = []

    for name, database_encodings in known_faces_db.items():
        distances = face_recognition.face_distance(
            database_encodings,
            current_encoding,
        )
        if len(distances):
            closest_distances = np.sort(distances)[:MATCH_SAMPLE_COUNT]
            robust_distance = float(np.mean(closest_distances))
            candidates.append((robust_distance, name, float(np.min(distances))))

    if not candidates:
        return "Unknown", None

    candidates.sort()
    best_distance, best_name, best_photo_distance = candidates[0]
    second_distance = candidates[1][0] if len(candidates) > 1 else 1.0

    print(
        "[MATCH] "
        + ", ".join(
            f"{name}: score={score:.3f}, best_photo={photo_distance:.3f}"
            for score, name, photo_distance in candidates[:3]
        )
    )

    if (
        best_distance > MATCH_THRESHOLD
        or second_distance - best_distance < MIN_MATCH_MARGIN
    ):
        return "Unknown", best_photo_distance

    return best_name, best_distance


def draw_detection(frame, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_origin = (x1, max(25, y1 - 8))
    cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def intersection_over_union(first_box, second_box):
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box
    overlap_x1 = max(first_x1, second_x1)
    overlap_y1 = max(first_y1, second_y1)
    overlap_x2 = min(first_x2, second_x2)
    overlap_y2 = min(first_y2, second_y2)
    overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union_area = first_area + second_area - overlap_area
    return overlap_area / union_area if union_area else 0.0


def find_track(person_box, tracks):
    best_track = None
    best_overlap = 0.3
    for track in tracks:
        overlap = intersection_over_union(person_box, track["bbox"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_track = track
    return best_track


def update_track_identity(track, candidate_name, candidate_distance):
    if candidate_name in ("Unknown", "No face") or track["identity_locked"]:
        return False

    track["votes"].append(candidate_name)
    vote_counts = Counter(track["votes"])
    voted_name, vote_count = vote_counts.most_common(1)[0]
    track["vote_count"] = vote_count
    track["vote_name"] = voted_name

    if vote_count >= MINIMUM_VOTES:
        track["name"] = voted_name
        track["confidence"] = candidate_distance
        track["identity_locked"] = True
        return True

    return False



def main():
    print(f"[TEST] Loading face database...")
    known_faces_db = load_database()
    print(f"[TEST] Loaded {len(known_faces_db)} people: {list(known_faces_db)}")

    print(f"[TEST] Loading YOLO model from {MODEL_PATH}...")
    yolo = YOLO(MODEL_PATH)
    source = get_stream_source()
    print(f"[TEST] Opening stream")

    stream = RTSPStream(source) if isinstance(source, str) else None
    capture = cv2.VideoCapture(source) if stream is None else None
    last_report = 0.0
    frame_number = 0
    tracks = []

    try:
        cv2.namedWindow("Neytra CCTV Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Neytra CCTV Test", 900, 700)
        while True:
            frame = stream.read() if stream is not None else capture.read()[1]
            if frame is None:
                print("[TEST] Waiting for a frame...")
                time.sleep(0.1)
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame_number += 1
            annotated = frame.copy()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            person_boxes = []
            results = yolo(frame, classes=[0], verbose=False)

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    confidence = float(box.conf[0])
                    if confidence < PERSON_CONFIDENCE:
                        continue
                    bbox = tuple(map(int, box.xyxy[0]))
                    if is_in_zone(bbox, ZONE_POINTS):
                        person_boxes.append(bbox)
                    else:
                        draw_detection(annotated, bbox, f"Person {confidence:.2f}", (255, 200, 0))

            frame_results = []
            for person_box in person_boxes:
                track = find_track(person_box, tracks)
                if track is None:
                    track = {
                        "bbox": person_box,
                        "name": "Recognizing...",
                        "confidence": None,
                        "votes": deque(maxlen=VOTE_WINDOW_SIZE),
                        "identity_locked": False,
                        "vote_name": None,
                        "vote_count": 0,
                        "last_recognition_frame": -RECOGNITION_INTERVAL,
                        "last_seen_frame": frame_number,
                    }
                    tracks.append(track)
                else:
                    track["bbox"] = person_box
                    track["last_seen_frame"] = frame_number

                if frame_number - track["last_recognition_frame"] >= RECOGNITION_INTERVAL:
                    candidate_name, candidate_distance = match_face(
                        image_rgb, person_box, known_faces_db
                    )
                    identity_locked_now = update_track_identity(
                        track,
                        candidate_name,
                        candidate_distance,
                    )
                    if identity_locked_now:
                        print(
                            f"[ANNOUNCE] {track['name']} locked "
                            f"after {MINIMUM_VOTES}/{VOTE_WINDOW_SIZE} votes "
                            f"(distance={track['confidence']:.3f})"
                        )
                    track["last_recognition_frame"] = frame_number

                name = track["name"]
                confidence = track["confidence"]
                frame_results.append((name, confidence, person_box))
                color = (0, 200, 0) if name not in ("Unknown", "No face") else (0, 0, 255)
                if name == "Recognizing..." and track["vote_name"]:
                    label = (
                        f"Checking {track['vote_name']} "
                        f"({track['vote_count']}/{MINIMUM_VOTES})"
                    )
                else:
                    label = name if confidence is None else f"{name} d={confidence:.3f}"
                draw_detection(annotated, person_box, label, color)

            tracks = [
                track for track in tracks
                if frame_number - track["last_seen_frame"] <= TRACK_MAX_MISSED_FRAMES
            ]

            cv2.polylines(annotated, [ZONE_POINTS], True, (255, 0, 255), 2)
            cv2.putText(
                annotated,
                f"People in zone: {len(person_boxes)} | q/Esc to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.polylines(annotated, [ZONE_POINTS], True, (255, 0, 255), 2)

            display_frame = cv2.resize(
                annotated,
                None,
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow("Neytra CCTV Test", display_frame)

            now = time.time()
            if frame_results and now - last_report >= 0.25:
                print(f"[RESULT] {len(frame_results)} person(s) in zone")
                for index, (name, confidence, _) in enumerate(frame_results, 1):
                    score = "n/a" if confidence is None else f"{confidence:.3f}"
                    print(f"  #{index}: {name} | distance={score}")
                last_report = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if stream is not None:
            stream.stop()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        print("[TEST] Stopped")


if __name__ == "__main__":
    main()
