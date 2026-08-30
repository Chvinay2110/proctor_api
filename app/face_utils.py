import numpy as np
from deepface import DeepFace

MODEL_NAME = "ArcFace"
DETECTOR = 'fastmtcnn'
THRESHOLD = 0.45   # kept for reference / potential distance-based checks


# ---- helper functions ----

def get_face_embedding(frame):
    if isinstance(frame, np.ndarray):
        reps = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True,
        )
        if not reps:
            raise ValueError("No face detected.")
        best = max(
            reps,
            key=lambda r: r.get("facial_area", {}).get("w", 0)
            * r.get("facial_area", {}).get("h", 0),
        )
        return np.array(best["embedding"], dtype=np.float64)
    else:
        return []


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.sqrt(np.sum(b ** 2))
    if denom == 0:
        return 1.0
    dist = float(1.0 - np.dot(a, b) / denom)
    return float(np.clip(dist, 0.0, 2.0))


def warm_up():
    """Force DeepFace to build and cache the detector + recognition models.

    DeepFace loads model weights lazily on first use, which costs several
    seconds. Calling this once at process startup (see main.py's lifespan
    handler) means that cost is paid before any candidate connects, instead
    of being paid by whichever candidate happens to connect first.
    """
    dummy_frame = np.zeros((160, 160, 3), dtype=np.uint8)

    try:
        DeepFace.build_model(MODEL_NAME)
        print(f"Warm-up: {MODEL_NAME} recognition model loaded.")
    except Exception as e:
        print(f"Warm-up: failed to build recognition model {MODEL_NAME}: {e}")

    try:
        # enforce_detection=False so a blank dummy frame (no real face) doesn't
        # raise -- the goal here is just to force the detector backend's
        # weights to load and be cached, not to detect anything real.
        DeepFace.extract_faces(img_path=dummy_frame, detector_backend=DETECTOR, enforce_detection=False)
        print(f"Warm-up: {DETECTOR} detector backend loaded.")
    except Exception as e:
        print(f"Warm-up: failed to build detector backend {DETECTOR}: {e}")


def face_match_percentage(frame, reference_embedding):
    curr_face_embedding = get_face_embedding(frame)
    dist = cosine_distance(curr_face_embedding, reference_embedding)
    similarity = round((1.0 - dist) * 100, 2)
    print(similarity)
    return similarity
