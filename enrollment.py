# Used only when want to change the Faces db, or enroll some new person
"""
enrollment.py

Batch-enrolls family members from photo folders into the face database,
reusing the exact DB format/location logic from recognizer.py.

Expected folder structure:

    known_faces/
        Mom/
            photo1.jpg
            photo2.jpg
        Dad/
            photo1.jpg
        You/
            photo1.jpg
            photo2.jpg
            photo3.jpg

Multiple photos per person is encouraged (different angles/lighting) -
each one adds another encoding to that person's list, which is exactly
how recognizer.py's compare_face_to_db already expects the data (it
takes the MIN distance across all of a person's stored encodings).

Uses YOLO for robust face detection on CCTV footage (handles angles, small faces, low quality).
Falls back to face_recognition if YOLO detection fails for cleaner studio photos.

Usage:
    python enroll_from_photos.py known_faces/
"""

import os
import sys
import cv2
import numpy as np
import face_recognition

# Reuse the exact same DB load/save logic and path resolution as the rest
# of the pipeline, so this writes to the same faces.pkl the recognizer reads.
from Neytra.Face.backend.ai.recognizer import load_database, save_database, DB_PATH

# DB_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "db",
#     "faces.pkl"
# )

# Try to load YOLO for robust CCTV face detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed - falling back to basic face detection")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
YOLO_MODEL = None


def load_yolo_model():
    """Load YOLO model for robust face detection on CCTV footage"""
    global YOLO_MODEL
    if not YOLO_AVAILABLE or YOLO_MODEL is not None:
        return YOLO_MODEL
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "models", "yolo11n.pt"))
        print(f"[YOLO] Loading from: {model_path}")
        YOLO_MODEL = YOLO(model_path)
        print("[YOLO] ✅ Loaded successfully for robust CCTV face detection")
        return YOLO_MODEL
    except Exception as e:
        print(f"[YOLO] ⚠️  Could not load: {e}")
        return None


def detect_faces_yolo(image_cv2):
    """
    Detect faces using YOLO (much better for CCTV angles/small faces).
    Returns list of (x1, y1, x2, y2) bounding boxes.
    
    Strategy: filter out background faces by:
    - Requiring reasonable confidence
    - Keeping only the largest face (most prominent in frame)
    """
    if YOLO_MODEL is None:
        return []
    
    try:
        results = YOLO_MODEL(image_cv2, imgsz=640, verbose=False)
        faces = []
        if results and len(results) > 0:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.5
                    # Only keep detections with reasonable confidence
                    if conf >= 0.3:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        faces.append((x1, y1, x2, y2, area, conf))
        
        # Return only the largest face (most prominent)
        if faces:
            faces.sort(key=lambda f: f[4], reverse=True)  # Sort by area (largest first)
            largest = faces[0]
            return [(largest[0], largest[1], largest[2], largest[3])]  # Return as (x1,y1,x2,y2)
        
        return []
    except Exception as e:
        print(f"  [YOLO Error] {e}")
        return []


def detect_faces_fallback(image):
    """
    Fallback: use face_recognition for clean studio photos.
    Returns list of (top, right, bottom, left) format (face_recognition standard).
    """
    try:
        face_locations = face_recognition.face_locations(image)
        return face_locations
    except Exception as e:
        print(f"  [Face Recognition Error] {e}")
        return []


def encode_photo(photo_path):
    """
    Load one photo and return its face encoding.
    Tries YOLO first (good for CCTV), falls back to face_recognition (good for studio photos).
    Returns None (with a printed reason) if it can't get exactly one clear face.

    # update : yolo detects person and reduces its bounding box to top 35%,
        which is then sent to dlib for face detection and encoding
    """
    try:
        image_cv2 = cv2.imread(photo_path)
        if image_cv2 is None:
            print(f"  [SKIP] Could not read file: {photo_path}")
            return None

        image_cv2 = cv2.rotate(image_cv2, cv2.ROTATE_90_CLOCKWISE)
        
        # Convert BGR (OpenCV) to RGB (face_recognition / dlib)
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # 1. Run YOLO to get person bounding boxes
        results = YOLO_MODEL(image_cv2, imgsz=640, classes=[0], verbose=False)
        face_locations = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Search for the actual face in the upper half of the person box.
                    h = y2 - y1
                    top = max(0, y1)
                    bottom = min(image_cv2.shape[0], y1 + int(h * 0.5))
                    left = max(0, x1)
                    right = min(image_cv2.shape[1], x2)

                    head_crop = np.ascontiguousarray(
                        image_rgb[top:bottom, left:right],
                        dtype=np.uint8,
                    )
                    detected_faces = face_recognition.face_locations(head_crop, model="hog")
                    if detected_faces:
                        largest_face = max(
                            detected_faces,
                            key=lambda location: (location[2] - location[0]) * (location[1] - location[3]),
                        )
                        face_top, face_right, face_bottom, face_left = largest_face
                        face_locations.append((
                            top + face_top,
                            left + face_right,
                            top + face_bottom,
                            left + face_left,
                        ))
                    else:
                        # Fallback for very small or blurred CCTV faces.
                        face_locations.append((top, right, min(
                            image_cv2.shape[0], y1 + int(h * 0.35)
                        ), left))

        # Fallback to dlib HOG if YOLO didn't detect anyone
        if not face_locations:
            face_locations = face_recognition.face_locations(image_rgb)

        # 2. Strict Enrollment Check: Must be EXACTLY 1 person
        if len(face_locations) == 0:
            print(f"  [SKIP] No person detected in {photo_path}")
            return None

        if len(face_locations) > 1:
            print(f"  [SKIP] {len(face_locations)} people detected in {photo_path}. Please use single-person photos for enrollment.")
            return None

        # 3. Generate 128-D vector using dlib landmark alignment on the YOLO box
        encodings = face_recognition.face_encodings(image_rgb, known_face_locations=face_locations)
        
        if encodings:
            print(f"  [OK] Valid 128-D encoding generated")
            return encodings[0]
        else:
            print(f"  [SKIP] dlib landmark predictor failed inside bounding box for {photo_path}")
            return None

    except Exception as e:
        print(f"  [ERROR] Processing {photo_path}: {e}")
        return None


def enroll_folder(root_folder):
    """
    Walk known_faces/<name>/*.jpg and build up the database.
    Uses YOLO for robust CCTV face detection + fallback to face_recognition.
    """
    if not os.path.isdir(root_folder):
        print(f"[ERROR] Folder not found: {root_folder}")
        sys.exit(1)

    # Load YOLO model once at startup for better CCTV detection
    if YOLO_AVAILABLE:
        load_yolo_model()
    
    db = {}
    print("[DB] Rebuilding database from enrollment photos")

    for person_name in sorted(os.listdir(root_folder)):
        person_dir = os.path.join(root_folder, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n[ENROLL] Processing {person_name}...")
        new_encodings = []

        for filename in sorted(os.listdir(person_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            photo_path = os.path.join(person_dir, filename)
            encoding = encode_photo(photo_path)
            if encoding is not None:
                new_encodings.append(encoding)

        if not new_encodings:
            print(f"  [WARN] No usable photos found for {person_name}, skipping")
            continue

        db[person_name] = new_encodings
        print(f"  [DONE] {person_name}: {len(new_encodings)} new encodings "
              f"({len(db[person_name])} total)")

    save_database(db)
    print(f"\n[SAVED] Database written to {DB_PATH}")
    print(f"[SAVED] Final DB: {[(name, len(encs)) for name, encs in db.items()]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python enroll_from_photos.py <path_to_known_faces_folder>")
        sys.exit(1)

    enroll_folder(sys.argv[1])