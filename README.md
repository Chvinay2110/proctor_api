# Face Verification API

FastAPI wrapper around your DeepFace (ArcFace + fastmtcnn) matching logic.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Models warm up on startup (see `warm_up()` in `face_utils.py`), so the first
real request isn't stuck paying that cost.

## Endpoint

`POST /verify-face`

### Request body

```json
{
  "images": [
    "<base64 reference image>",
    "<base64 frame 1>",
    "<base64 frame 2>",
    "<base64 frame 3>",
    "<base64 frame 4>",
    "<base64 frame 5>",
    "<base64 frame 6>",
    "<base64 frame 7>"
  ]
}
```

- Must contain exactly 8 entries: `images[0]` is the reference face,
  `images[1:]` are the 7 frames to check against it.
- Each string can be raw base64 or a data-URI (`data:image/jpeg;base64,...`).

### Response body

```json
{
  "verified": true,
  "frames_passed": 6,
  "frames_total": 7,
  "min_required": 5,
  "threshold_percent": 70.0,
  "results": [
    {"frame_index": 1, "similarity": 91.2, "passed": true, "error": null},
    {"frame_index": 2, "similarity": 88.4, "passed": true, "error": null},
    {"frame_index": 3, "similarity": null, "passed": false, "error": "No face detected."},
    ...
  ]
}
```

- `verified` is `true` only if `frames_passed >= min_required` (5 of 7 by default).
- A frame that errors out (no face detected, bad image data, etc.) counts as
  a non-passing frame rather than failing the whole request -- one bad
  frame in the middle of a burst shouldn't tank the entire check.
- If the **reference** image itself has no detectable face, the endpoint
  returns `400` instead, since there's nothing to compare against.

## Design notes

- `frames_passed`, `min_required`, `threshold_percent` are all in the
  response so the frontend doesn't have to hardcode "5 of 7" anywhere --
  the API is the single source of truth for the pass policy.
- Tunable constants (`EXPECTED_TOTAL_IMAGES`, `MIN_PASSING_FRAMES`,
  `MATCH_THRESHOLD_PERCENT`) live in `app/schemas.py` if you want to change
  the policy later (e.g. loosen to 4/7, or make it configurable via env vars).
- `DeepFace.represent` / `extract_faces` calls are blocking CPU work, so
  they're run via `run_in_threadpool` rather than directly in the async
  route -- otherwise one slow verification would stall every other request
  hitting this service at the same time.
- Reference embedding is computed once and reused for all 7 comparisons
  instead of re-embedding it per frame.

## Note on the `fastmtcnn` detector

DeepFace's `fastmtcnn` backend depends on `facenet-pytorch`, which is
included in `requirements.txt`. First import will also pull in `torch` if
it's not already installed.
