import base64
import re

import cv2
import numpy as np

# strips a data-URI prefix like "data:image/jpeg;base64," if present
_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,")


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image string into a BGR numpy array (OpenCV format).

    Accepts both raw base64 and data-URI style strings
    (e.g. 'data:image/jpeg;base64,...'). Raises ValueError on any failure
    so callers can turn it into a clean per-frame error instead of a 500.
    """
    if not b64_string or not isinstance(b64_string, str):
        raise ValueError("Empty or invalid base64 string.")

    cleaned = _DATA_URI_RE.sub("", b64_string.strip())

    try:
        img_bytes = base64.b64decode(cleaned, validate=False)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {e}")

    if not img_bytes:
        raise ValueError("Decoded base64 payload was empty.")

    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image bytes into a valid image.")

    return frame
