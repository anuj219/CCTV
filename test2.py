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

from cctv.ai_summary import generate_scene_summary, should_generate_summary
from cctv.db import create_entry_event, init_db, should_log_entry, update_entry_event
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
RECHECK_INTERVAL_SECONDS = 10.0
RECOGNITION_RETRY_SECONDS = 1.0
START_LINE_Y = 1000
DOOR_LINE_Y = 1900
ENTRY_COOLDOWN_SECONDS = 60.0
UNKNOWN_LINGER_SECONDS = 15.0
TRACK_MAX_MISSED_FRAMES = 30
ROTATION = cv2.ROTATE_90_CLOCKWISE
MODEL_PATH = os.path.join("models", "yolo11n.pt")
KNOWN_FACES_PATH = os.path.join("cctv", "known_faces")
GALLERY_PATH = os.path.join("cctv", "arcface_gallery.pkl")
ZONE_POINTS = np.array(
    [[100, 1100], [800, 600], [800, 2000], [100, 2000]],
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
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame,
        label,
        (x1, max(40, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA,
    )


def update_entry_state(track, person_box):
    _, _, _, y2 = person_box
    current_foot_y = y2
    previous_foot_y = track["previous_foot_y"]
    track["previous_foot_y"] = current_foot_y

    if previous_foot_y is None:
        return

    crossed_start = previous_foot_y < START_LINE_Y <= current_foot_y
    crossed_door = previous_foot_y < DOOR_LINE_Y <= current_foot_y
    crossed_both = previous_foot_y < START_LINE_Y and current_foot_y >= DOOR_LINE_Y

    if crossed_start:
        track["has_approached"] = True
        print(
            f"[CROSSING] START line crossed: foot_y "
            f"{previous_foot_y}->{current_foot_y}"
        )

    if crossed_door and track["has_approached"] or crossed_both:
        track["entry_pending"] = True
        track["entry_crossed_at"] = time.monotonic()
        print(
            f"[CROSSING] DOOR line crossed: foot_y "
            f"{previous_foot_y}->{current_foot_y}"
        )


def update_unknown_linger_state(track, now, headcount, frame=None):
    if not track.get("has_approached", False):
        track["unknown_linger_started_at"] = None
        return

    if track.get("locked", False):
        track["unknown_linger_started_at"] = None
        return

    is_unknown = (
        track.get("unknown_confirmed")
        or track.get("entry_status") in {"unconfirmed_unknown", "confirmed_unknown"}
        or track.get("name") == "Recognizing..."
    )

    if not is_unknown:
        track["unknown_linger_started_at"] = None
        return

    if track.get("unknown_linger_started_at") is None:
        track["unknown_linger_started_at"] = now
        print(
            f"[AI SUMMARY] Unknown / no-face visitor in door zone; "
            f"linger timer started at {now:.1f}s"
        )
        return

    if (now - float(track["unknown_linger_started_at"])) < UNKNOWN_LINGER_SECONDS:
        return

    if track.get("entry_event_id") is None:
        track["entry_event_id"] = create_entry_event(
            "confirmed_unknown",
            headcount,
            track["track_id"],
        )
    update_entry_event(
        track["entry_event_id"],
        "confirmed_unknown",
        "Unknown",
    )
    track["entry_status"] = "confirmed_unknown"
    track["unknown_confirmed"] = True

    if should_generate_summary(track, now):
        summary = generate_scene_summary(
            track["entry_event_id"],
            frame=frame,
            headcount=headcount,
            reason="unknown_linger",
        )
        track["last_ai_summary_at"] = now
        if summary:
            print(f"[AI SUMMARY] {summary}")


def emit_entry_if_ready(track, now, headcount, frame=None):
    if not track["entry_pending"]:
        return

    if track["locked"]:
        person_name = track["name"]
        allowed = should_log_entry(person_name, ENTRY_COOLDOWN_SECONDS, track["entry_event_id"])
        status = "known" if allowed else "known_suppressed"
        if track["entry_event_id"] is None:
            track["entry_event_id"] = create_entry_event(
                status,
                headcount,
                track["track_id"],
            )
        update_entry_event(
            track["entry_event_id"],
            status,
            person_name,
            track["score"],
        )
        if not allowed:
            print(
                f"[ENTRY] Suppressed {person_name}: "
                f"database cooldown active ({ENTRY_COOLDOWN_SECONDS:.0f}s)"
            )
        else:
            if track["entry_status"] == "unconfirmed_unknown":
                print(f"[ENTRY UPDATE] Previously unconfirmed entry resolved as {person_name}")
            else:
                print(f"[ENTRY] {person_name} crossed the doorway")
        track["entry_status"] = status
        track["entry_pending"] = False
        return

    if track["unknown_confirmed"]:
        if track["entry_event_id"] is None:
            track["entry_event_id"] = create_entry_event(
                "confirmed_unknown",
                headcount,
                track["track_id"],
            )
        update_entry_event(
            track["entry_event_id"],
            "confirmed_unknown",
            "Unknown",
        )
        if track["entry_status"] == "unconfirmed_unknown":
            print("[ENTRY UPDATE] Previously unconfirmed entry confirmed as unknown visitor")
        else:
            print("[ENTRY] Confirmed unknown visitor crossed the doorway")
        if should_generate_summary(track, now):
            summary = generate_scene_summary(
                track["entry_event_id"],
                frame=frame,
                headcount=headcount,
                reason="unknown_entry",
            )
            track["last_ai_summary_at"] = now
            if summary:
                print(f"[AI SUMMARY] {summary}")
        track["entry_status"] = "confirmed_unknown"
        track["entry_pending"] = False
        return

    if track["entry_status"] is None:
        track["entry_event_id"] = create_entry_event(
            "unconfirmed_unknown",
            headcount,
            track["track_id"],
        )
        track["entry_status"] = "unconfirmed_unknown"
        print("[ENTRY] Unconfirmed unknown visitor crossed the doorway")
        if should_generate_summary(track, now):
            summary = generate_scene_summary(
                track["entry_event_id"],
                frame=frame,
                headcount=headcount,
                reason="no_face",
            )
            track["last_ai_summary_at"] = now
            if summary:
                print(f"[AI SUMMARY] {summary}")


def recognition_is_due(track, frame_number, now):
    if now - track["last_attempt_at"] < RECOGNITION_RETRY_SECONDS:
        return False
    if not track["locked"]:
        return frame_number - track["last_recognition"] >= RECOGNITION_INTERVAL
    return now - track["last_confirmed_at"] >= RECHECK_INTERVAL_SECONDS


def main():
    init_db()
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
    next_track_id = 1
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
                             "locked": False, "last_seen": frame_number,
                             "last_recognition": -RECOGNITION_INTERVAL,
                             "last_attempt_at": 0.0, "last_confirmed_at": None,
                             "previous_foot_y": None, "has_approached": False,
                             "entry_pending": False, "entry_status": None,
                             "entry_crossed_at": None, "unknown_votes": deque(maxlen=VOTE_WINDOW_SIZE),
                             "unknown_confirmed": False, "entry_event_id": None,
                             "unknown_linger_started_at": None, "last_ai_summary_at": 0.0,
                             "last_frame": frame.copy(), "track_id": f"track-{next_track_id}"}
                    next_track_id += 1
                    tracks.append(track)
                else:
                    track["bbox"] = person_box
                    track["last_seen"] = frame_number
                track["last_frame"] = frame.copy()
                update_entry_state(track, person_box)

            now = time.monotonic()
            due_tracks = [
                track for track in tracks
                if track["last_seen"] == frame_number
                and recognition_is_due(track, frame_number, now)
            ]
            faces = []
            if due_tracks:
                face_started_at = time.perf_counter()
                faces = [face for face in face_app.get(frame) if is_good_face(face)]
                face_elapsed_ms = (time.perf_counter() - face_started_at) * 1000
                print(
                    f"[TIMING] ArcFace={face_elapsed_ms:.0f}ms "
                    f"tracks={len(due_tracks)}"
                )

            for track in due_tracks:
                person_box = track["bbox"]
                track["last_attempt_at"] = now
                track["last_recognition"] = frame_number
                face = face_for_person(faces, person_box)
                if face is not None:
                    current_embedding = normalize_embedding(face.embedding)
                    if track["locked"]:
                        pooled_embedding = current_embedding
                    else:
                        track["embeddings"].append(current_embedding)
                        pooled_embedding = normalize_embedding(np.mean(track["embeddings"], axis=0))

                    candidate_name, candidate_score = match_embedding(pooled_embedding, gallery)
                    if track["locked"]:
                        if candidate_name == track["name"]:
                            track["score"] = candidate_score
                            track["last_confirmed_at"] = now
                        else:
                            print(
                                f"[ARCFACE] Recheck failed for {track['name']}; "
                                "unlocking track"
                            )
                            track["locked"] = False
                            track["name"] = "Recognizing..."
                            track["score"] = None
                            track["votes"].clear()
                            track["embeddings"].clear()
                            track["unknown_votes"].clear()
                            track["unknown_confirmed"] = False
                    elif candidate_name != "Unknown":
                        track["unknown_votes"].clear()
                        track["unknown_confirmed"] = False
                        track["votes"].append(candidate_name)
                        voted_name, vote_count = Counter(track["votes"]).most_common(1)[0]
                        if vote_count >= MINIMUM_VOTES:
                            track["name"] = voted_name
                            track["score"] = candidate_score
                            track["locked"] = True
                            track["last_confirmed_at"] = now
                            print(
                                f"[ARCFACE ANNOUNCE] {voted_name} locked "
                                f"after {vote_count}/{VOTE_WINDOW_SIZE} votes"
                            )
                    elif candidate_score is not None:
                        track["votes"].clear()
                        track["unknown_votes"].append(True)
                        if (
                            not track["unknown_confirmed"]
                            and sum(track["unknown_votes"]) >= MINIMUM_VOTES
                        ):
                            track["unknown_confirmed"] = True
                            print(
                                "[ARCFACE] Visitor confirmed unknown "
                                f"after {sum(track['unknown_votes'])}/{VOTE_WINDOW_SIZE} samples"
                            )

                update_unknown_linger_state(track, now, len(person_boxes), frame)
                emit_entry_if_ready(track, now, len(person_boxes), frame)

            for person_box in person_boxes:
                track = find_track(person_box, tracks)
                if track is None:
                    continue

                emit_entry_if_ready(track, now, len(person_boxes), frame)

                label = track["name"]
                if label == "Recognizing..." and track["votes"]:
                    label = f"Checking {track['votes'][-1]} ({track['votes'].count(track['votes'][-1])}/{MINIMUM_VOTES})"
                elif track["score"] is not None:
                    label = f"{label} d={track['score']:.3f}"
                draw_detection(annotated, person_box, label, (0, 200, 0) if track["locked"] else (0, 165, 255))

            tracks[:] = [track for track in tracks if frame_number - track["last_seen"] <= TRACK_MAX_MISSED_FRAMES]
            cv2.polylines(annotated, [ZONE_POINTS], True, (255, 0, 255), 2)
            frame_width = annotated.shape[1]
            cv2.line(
                annotated,
                (0, START_LINE_Y),
                (frame_width, START_LINE_Y),
                (0, 255, 255),
                3,
            )
            cv2.line(
                annotated,
                (0, DOOR_LINE_Y),
                (frame_width, DOOR_LINE_Y),
                (0, 0, 255),
                3,
            )
            cv2.putText(
                annotated,
                "START",
                (20, START_LINE_Y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "DOOR",
                (20, DOOR_LINE_Y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"People in zone: {len(person_boxes)} | q/Esc to quit",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
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
