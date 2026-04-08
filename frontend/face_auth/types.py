from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class FaceAuthStage(str, Enum):
    IDLE = "IDLE"
    MATCHING = "MATCHING"
    LIVENESS = "LIVENESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ChallengeType(str, Enum):
    BLINK = "BLINK"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    OPEN_MOUTH = "OPEN_MOUTH"


class LivenessState(str, Enum):
    WAIT_CAMERA = "WAIT_CAMERA"
    WAIT_FACE = "WAIT_FACE"
    ALIGNING = "ALIGNING"
    READY = "READY"
    CHALLENGE_ACTIVE = "CHALLENGE_ACTIVE"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


CHALLENGE_INSTRUCTIONS = {
    ChallengeType.BLINK: "Blink once",
    ChallengeType.TURN_LEFT: "Turn your head left",
    ChallengeType.TURN_RIGHT: "Turn your head right",
    ChallengeType.OPEN_MOUTH: "Open your mouth",
}


CHALLENGE_LABELS_KO = {
    ChallengeType.BLINK: "눈을 한 번 깜빡이세요",
    ChallengeType.TURN_LEFT: "고개를 왼쪽으로 돌리세요",
    ChallengeType.TURN_RIGHT: "고개를 오른쪽으로 돌리세요",
    ChallengeType.OPEN_MOUTH: "입을 벌리세요",
}


LIVENESS_STATE_LABELS_KO = {
    LivenessState.WAIT_CAMERA: "카메라 권한 대기",
    LivenessState.WAIT_FACE: "얼굴 인식 대기",
    LivenessState.ALIGNING: "얼굴 정렬 중",
    LivenessState.READY: "시작 가능",
    LivenessState.CHALLENGE_ACTIVE: "지령 수행 중",
    LivenessState.SUCCESS: "인증 통과",
    LivenessState.FAILED: "인증 실패",
}


@dataclass(slots=True)
class FrameAnalysis:
    face_detected: bool
    single_face: bool
    center_ok: bool
    size_ok: bool
    stable_ok: bool
    left_ear: float
    right_ear: float
    mouth_ratio: float
    nose_ratio: float
    guide_text: str
    face_count: int = 0
    front_facing: bool = False
    stable_frame_count: int = 0
    frame_width: int = 0
    frame_height: int = 0
    face_box: Optional[tuple[int, int, int, int]] = None
    face_signature: Optional[np.ndarray] = None

    @property
    def average_ear(self) -> float:
        if self.left_ear <= 0.0 and self.right_ear <= 0.0:
            return 0.0
        return (self.left_ear + self.right_ear) / 2.0


@dataclass(slots=True)
class LivenessSnapshot:
    state: LivenessState
    guide_text: str
    instruction_text: str
    current_challenge: Optional[ChallengeType] = None
    pending_challenges: list[ChallengeType] = field(default_factory=list)
    completed_challenges: list[ChallengeType] = field(default_factory=list)
    retries_used: int = 0
    max_retries: int = 0
    failure_reason: Optional[str] = None
    ready_for_start: bool = False
    replacement_required: bool = False
    analysis: Optional[FrameAnalysis] = None


@dataclass(slots=True)
class FaceMatchResult:
    matched: bool
    score: float
    threshold: float


@dataclass(slots=True)
class FaceAuthSnapshot:
    stage: FaceAuthStage
    guide_text: str
    badge_text: str
    instruction_text: str = ""
    match_score: Optional[float] = None
    match_threshold: float = 0.0
    result_message: str = ""
    liveness_snapshot: Optional[LivenessSnapshot] = None
