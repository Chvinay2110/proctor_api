from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# --- tunable policy constants ---
EXPECTED_TOTAL_IMAGES = 8            # 1 reference + 7 verification frames
REQUIRED_FRAMES = EXPECTED_TOTAL_IMAGES - 1
MIN_PASSING_FRAMES = 5               # need >= 5 of 7 frames to count as verified
MATCH_THRESHOLD_PERCENT = 70.0       # a single frame "passes" at >= 70% similarity


class FaceVerificationRequest(BaseModel):
    images: List[str] = Field(
        ...,
        description=(
            "List of base64-encoded images (raw base64 or data-URI). "
            "images[0] is the reference image; images[1:] are the frames "
            "to verify against it. Must contain exactly "
            f"{EXPECTED_TOTAL_IMAGES} entries."
        ),
    )

    @field_validator("images")
    @classmethod
    def validate_length(cls, v: List[str]) -> List[str]:
        if len(v) != EXPECTED_TOTAL_IMAGES:
            raise ValueError(
                f"Expected exactly {EXPECTED_TOTAL_IMAGES} images "
                f"(1 reference + {REQUIRED_FRAMES} frames), got {len(v)}."
            )
        return v


class FrameResult(BaseModel):
    frame_index: int                       # 1-based index among the verification frames
    similarity: Optional[float] = None     # None if the frame errored out (e.g. no face)
    passed: bool
    error: Optional[str] = None


class FaceVerificationResponse(BaseModel):
    verified: bool
    frames_passed: int
    frames_total: int
    min_required: int
    threshold_percent: float
    results: List[FrameResult]
