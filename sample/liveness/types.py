# 프로젝트의 공용 데이터 구조를 정의
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# 어떤 지령인지
class ChallengeType(str, Enum):
    BLINK = "BLINK"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    OPEN_MOUTH = "OPEN_MOUTH"


# 현재 인증 단계가 무엇인지
class SessionState(str, Enum):
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


# 한 프레임을 분석한 결과
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

    @property
    def average_ear(self) -> float:
        if self.left_ear <= 0.0 and self.right_ear <= 0.0:
            return 0.0
        return (self.left_ear + self.right_ear) / 2.0


# 현재 세션 전체 상태
@dataclass(slots=True)
class SessionSnapshot:
    state: SessionState
    guide_text: str
    instruction_text: str
    current_challenge: Optional[ChallengeType] = None
    pending_challenges: list[ChallengeType] = field(default_factory=list)
    completed_challenges: list[ChallengeType] = field(default_factory=list)
    retries_used: int = 0
    max_retries: int = 2
    failure_reason: Optional[str] = None
    ready_for_start: bool = False
    replacement_required: bool = False
    analysis: Optional[FrameAnalysis] = None
