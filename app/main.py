import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from .face_utils import face_match_percentage, get_face_embedding, warm_up
from .image_utils import decode_base64_image
from .schemas import (
    MATCH_THRESHOLD_PERCENT,
    MIN_PASSING_FRAMES,
    FaceVerificationRequest,
    FaceVerificationResponse,
    FrameResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_verify")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # DeepFace lazy-loads model + detector weights on first call. Doing it
    # here means the first real request doesn't eat that latency.
    logger.info("Warming up DeepFace models...")
    await run_in_threadpool(warm_up)
    logger.info("Warm-up complete. Service ready.")
    yield


app = FastAPI(title="Face Verification Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/verify-face", response_model=FaceVerificationResponse)
async def verify_face(payload: FaceVerificationRequest):
    """
    Expects payload.images = [reference_b64, frame1_b64, ..., frame7_b64].

    Verifies the reference face against each of the 7 frames independently,
    then passes overall if at least MIN_PASSING_FRAMES (5) of the frames
    scored >= MATCH_THRESHOLD_PERCENT (60%) similarity.
    """
    reference_b64, *frame_b64_list = payload.images

    # --- reference image: decode + embed once, reuse for every frame ---
    try:
        reference_frame = decode_base64_image(reference_b64)
        # DeepFace calls are CPU-bound (numpy/opencv/onnx under the hood) and
        # blocking, so they're pushed to a threadpool to avoid stalling the
        # event loop while other requests are waiting.
        reference_embedding = await run_in_threadpool(get_face_embedding, reference_frame)
    except Exception as e:
        logger.warning(f"Reference image failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not process reference image: {e}",
        )

    # --- check each frame against the reference embedding ---
    results: list[FrameResult] = []
    frames_passed = 0

    for idx, b64_frame in enumerate(frame_b64_list, start=1):
        try:
            frame = decode_base64_image(b64_frame)
            similarity = await run_in_threadpool(
                face_match_percentage, frame, reference_embedding
            )
            passed = similarity >= MATCH_THRESHOLD_PERCENT
            if passed:
                frames_passed += 1
            results.append(
                FrameResult(frame_index=idx, similarity=similarity, passed=passed)
            )
        except Exception as e:
            # A single bad/no-face frame should not fail the whole request --
            # it just counts as a non-passing frame with an error attached.
            logger.info(f"Frame {idx} failed: {e}")
            results.append(
                FrameResult(frame_index=idx, similarity=None, passed=False, error=str(e))
            )

    verified = frames_passed >= MIN_PASSING_FRAMES

    return FaceVerificationResponse(
        verified=verified,
        frames_passed=frames_passed,
        frames_total=len(frame_b64_list),
        min_required=MIN_PASSING_FRAMES,
        threshold_percent=MATCH_THRESHOLD_PERCENT,
        results=results,
    )
