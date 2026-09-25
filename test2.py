"""ArcFace CCTV prototype.

Run from the project root:
    .venv/Scripts/python.exe -m cctv.test2

This is intentionally separate from cctv.test. It keeps YOLO person-zone
filtering and IoU tracking, but uses InsightFace/ArcFace for face detection,
alignment, and 512-D embeddings.
"""

import os
import pickle
import time
from collections import Counter, deque

import cv2
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from ultralytics import YOLO

from cctv.rtsp_stream import RTSPStream

load_dotenv()

PERSON_CONFIDENCE = 0.45
FACE_DETECTION_SIZE = (640, 640)
MIN_FACE_SIZE = 28
MIN_FACE_SHARPNESS = 35.0
ARCFACE_DISTANCE_THRESHOLD = 0.45
MIN_MATCH_MARGIN = 0.05
VOTE_WINDOW_SIZE = 3
MINIMUM_VOTES = 2
EMBEDDINGS_PER_TRACK = 5
RECOGNITION_INTERVAL = 3
TRACK_MAX_MISSED_FRAMES = 30
ROTATION = cv2.ROTATE_90_CLOCKWISE
MODEL_PATH = os.path.join("models", "yolo11n.pt")
KNOWN_FACES_PATH = os.path.join("cctv", "known_faces")
GALLERY_PATH = os.path.join("cctv", "arcface_gallery.pkl")
ZONE_POINTS = np.array(
    [[100, 1000], [800, 700], [800, 2100], [100, 2100]],
    np.int32,
)


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm else embedding


def image_quality(face):
    x1, y1, x2, y2 = map(int, face.bbox)
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return min(width, height), float(getattr(face, "det_score", 0.0))


def is_good_face(face):
    size, detection_score = image_quality(face)
    return size >= MIN_FACE_SIZE and detection_score >= 0.45


def build_gallery(face_app, root_folder):
    if os.path.exists(GALLERY_PATH):
        with open(GALLERY_PATH, "rb") as file_handle:
            gallery = pickle.load(file_handle)
        print(f"[ARCFACE] Loaded gallery from {GALLERY_PATH}: {list(gallery)}")
        return gallery

    gallery_vectors = {}
    for person_name in sorted(os.listdir(root_folder)):
        person_folder = os.path.join(root_folder, person_name)
        if not os.path.isdir(person_folder):
            continue

        vectors = []
        for filename in sorted(os.listdir(person_folder)):
            if os.path.splitext(filename)[1].lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = cv2.imread(os.path.join(person_folder, filename))
            if image is None:
                continue
            image = cv2.rotate(image, ROTATION)
            faces = [face for face in face_app.get(image) if is_good_face(face)]
            if not faces:
                print(f"[ARCFACE] Skip {person_name}/{filename}: no quality face")
                continue
            face = max(faces, key=lambda item: image_quality(item)[0])
            vectors.append(normalize_embedding(face.embedding))

        if vectors:
            matrix = np.asarray(vectors, dtype=np.float32)
            prototype = normalize_embedding(np.mean(matrix, axis=0))
            gallery_vectors[person_name] = {
                "prototype": prototype,
                "embeddings": matrix,
            }
            print(f"[ARCFACE] {person_name}: {len(vectors)} embeddings")

    with open(GALLERY_PATH, "wb") as file_handle:
        pickle.dump(gallery_vectors, file_handle)
    print(f"[ARCFACE] Saved gallery to {GALLERY_PATH}")
    return gallery_vectors


def is_in_zone(bbox):
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) // 2, y2)
    return cv2.pointPolygonTest(ZONE_POINTS, center, False) >= 0


def intersection_over_union(first_box, second_box):
    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first_box[2] - first_box[0]) * max(0, first_box[3] - first_box[1])
    second_area = max(0, second_box[2] - second_box[0]) * max(0, second_box[3] - second_box[1])
    union = first_area + second_area - overlap
    return overlap / union if union else 0.0


def find_track(person_box, tracks):
    matches = [(intersection_over_union(person_box, track["bbox"]), track) for track in tracks]
    matches = [match for match in matches if match[0] > 0.3]
    return max(matches, default=(0.0, None), key=lambda item: item[0])[1]


def face_for_person(faces, person_box):
    x1, y1, x2, y2 = person_box
    candidates = []
    for face in faces:
        fx1, fy1, fx2, fy2 = map(int, face.bbox)
        center_x = (fx1 + fx2) / 2
        center_y = (fy1 + fy2) / 2
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            candidates.append(face)
    return max(candidates, default=None, key=lambda face: image_quality(face)[0])


def match_embedding(embedding, gallery):
    embedding = normalize_embedding(embedding)
    candidates = []
    for name, person in gallery.items():
        prototype_distance = 1.0 - float(np.dot(person["prototype"], embedding))
        individual_distances = 1.0 - np.dot(person["embeddings"], embedding)
        best_individual_distance = float(np.min(individual_distances))
        score = 0.7 * prototype_distance + 0.3 * best_individual_distance
        candidates.append((score, name, prototype_distance, best_individual_distance))

    if not candidates:
        return "Unknown", None
    candidates.sort()
    best_score, best_name, _, _ = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 1.0
    print("\n-------------")
    print("[ARCFACE MATCH] " + ", ".join(
        f"{name}: score={score:.3f}, prototype={prototype:.3f}, best={best:.3f}"
        for score, name, prototype, best in candidates[:3]
    ))
    if best_score > ARCFACE_DISTANCE_THRESHOLD or second_score - best_score < MIN_MATCH_MARGIN:
        return "Unknown", best_score
    return best_name, best_score


def draw_detection(frame, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    print("[ARCFACE] Loading buffalo_l model; first run may download model files...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=FACE_DETECTION_SIZE)
    gallery = build_gallery(face_app, KNOWN_FACES_PATH)
    yolo = YOLO(MODEL_PATH)

    username = os.getenv("uname", "")
    password = os.getenv("password", "")
    source = os.getenv("CAMERA_URL") or os.getenv("RTSP_URL") or f"rtsp://{username}:{password}@192.168.29.62:554/Streaming/Channels/301"
    stream = RTSPStream(source) if isinstance(source, str) else None
    capture = cv2.VideoCapture(source) if stream is None else None
    tracks = []
    frame_number = 0
    try:
        cv2.namedWindow("Neytra CCTV ArcFace Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Neytra CCTV ArcFace Test", 900, 700)
        while True:
            frame = stream.read() if stream else capture.read()[1]
            if frame is None:
                time.sleep(0.1)
                continue
            frame = cv2.rotate(frame, ROTATION)
            frame_number += 1
            annotated = frame.copy()
            faces = [face for face in face_app.get(frame) if is_good_face(face)]
            person_boxes = []
            results = yolo(frame, classes=[0], verbose=False)
            for box in results[0].boxes if results and results[0].boxes is not None else []:
                if float(box.conf[0]) < PERSON_CONFIDENCE:
                    continue
                person_box = tuple(map(int, box.xyxy[0]))
                if is_in_zone(person_box):
                    person_boxes.append(person_box)
                else:
                    draw_detection(annotated, person_box, "Person", (255, 200, 0))

            for person_box in person_boxes:
                track = find_track(person_box, tracks)
                if track is None:
                    track = {"bbox": person_box, "name": "Recognizing...", "score": None,
                             "votes": deque(maxlen=VOTE_WINDOW_SIZE), "embeddings": deque(maxlen=EMBEDDINGS_PER_TRACK),
                             "locked": False, "last_seen": frame_number, "last_recognition": -RECOGNITION_INTERVAL}
                    tracks.append(track)
                else:
                    track["bbox"] = person_box
                    track["last_seen"] = frame_number

                if frame_number - track["last_recognition"] >= RECOGNITION_INTERVAL:
                    face = face_for_person(faces, person_box)
                    if face is not None:
                        track["embeddings"].append(normalize_embedding(face.embedding))
                        pooled_embedding = normalize_embedding(np.mean(track["embeddings"], axis=0))
                        candidate_name, candidate_score = match_embedding(pooled_embedding, gallery)
                        if not track["locked"] and candidate_name != "Unknown":
                            track["votes"].append(candidate_name)
                            voted_name, vote_count = Counter(track["votes"]).most_common(1)[0]
                            if vote_count >= MINIMUM_VOTES:
                                track["name"] = voted_name
                                track["score"] = candidate_score
                                track["locked"] = True
                                print(f"[ARCFACE ANNOUNCE] {voted_name} locked after {vote_count}/{VOTE_WINDOW_SIZE} votes")
                    track["last_recognition"] = frame_number

                label = track["name"]
                if label == "Recognizing..." and track["votes"]:
                    label = f"Checking {track['votes'][-1]} ({track['votes'].count(track['votes'][-1])}/{MINIMUM_VOTES})"
                elif track["score"] is not None:
                    label = f"{label} d={track['score']:.3f}"
                draw_detection(annotated, person_box, label, (0, 200, 0) if track["locked"] else (0, 165, 255))

            tracks[:] = [track for track in tracks if frame_number - track["last_seen"] <= TRACK_MAX_MISSED_FRAMES]
            cv2.polylines(annotated, [ZONE_POINTS], True, (255, 0, 255), 2)
            cv2.putText(annotated, f"People in zone: {len(person_boxes)} | q/Esc to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Neytra CCTV ArcFace Test", cv2.resize(annotated, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if stream:
            stream.stop()
        if capture:
            capture.release()
        cv2.destroyAllWindows()
        print("[ARCFACE] Stopped")


if __name__ == "__main__":
    main()
