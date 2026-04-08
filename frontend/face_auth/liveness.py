from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

from .types import CHALLENGE_INSTRUCTIONS, ChallengeType, FrameAnalysis, LivenessSnapshot, LivenessState


@dataclass(slots=True)
class BaselineMetrics:
    average_ear: float
    mouth_ratio: float
    nose_ratio: float


class BaseChallengeDetector:
    def __init__(self, baseline: BaselineMetrics) -> None:
        self.baseline = baseline
        self.success = False
        self.wrong_action = False
        self.feedback = "Waiting for action"

    def update(self, analysis: FrameAnalysis) -> bool:
        raise NotImplementedError


class BlinkDetector(BaseChallengeDetector):
    def __init__(self, baseline: BaselineMetrics) -> None:
        super().__init__(baseline)
        self.closed_frames = 0
        self.closed_detected = False

    def update(self, analysis: FrameAnalysis) -> bool:
        average_ear = analysis.average_ear
        close_threshold = min(0.20, self.baseline.average_ear * 0.72)
        reopen_threshold = max(0.23, self.baseline.average_ear * 0.88)

        if average_ear > 0 and average_ear < close_threshold:
            self.closed_frames += 1
            self.feedback = "Eyes closed"
        else:
            if self.closed_frames >= 2:
                self.closed_detected = True
            self.closed_frames = 0
            self.feedback = "Blink detection in progress"

        if self.closed_detected and average_ear > reopen_threshold:
            self.success = True
            self.feedback = "Blink detected"

        return self.success


class OpenMouthDetector(BaseChallengeDetector):
    def __init__(self, baseline: BaselineMetrics) -> None:
        super().__init__(baseline)
        self.open_frames = 0

    def update(self, analysis: FrameAnalysis) -> bool:
        open_threshold = max(0.06, self.baseline.mouth_ratio + 0.02)
        if analysis.mouth_ratio > open_threshold:
            self.open_frames += 1
            self.feedback = "Mouth opening detected"
        else:
            self.open_frames = 0
            self.feedback = "Open your mouth wider"

        if self.open_frames >= 5:
            self.success = True
            self.feedback = "Mouth opening detected"

        return self.success


class TurnLeftDetector(BaseChallengeDetector):
    def __init__(self, baseline: BaselineMetrics) -> None:
        super().__init__(baseline)
        self.left_frames = 0
        self.wrong_frames = 0

    def update(self, analysis: FrameAnalysis) -> bool:
        if analysis.nose_ratio < 0.42:
            self.left_frames += 1
            self.feedback = "Left turn detected"
        else:
            self.left_frames = 0
            self.feedback = "Turn your head left"

        if analysis.nose_ratio > 0.58:
            self.wrong_frames += 1
        else:
            self.wrong_frames = 0

        if self.left_frames >= 5:
            self.success = True
        if self.wrong_frames >= 5:
            self.wrong_action = True
            self.feedback = "Wrong direction"

        return self.success


class TurnRightDetector(BaseChallengeDetector):
    def __init__(self, baseline: BaselineMetrics) -> None:
        super().__init__(baseline)
        self.right_frames = 0
        self.wrong_frames = 0

    def update(self, analysis: FrameAnalysis) -> bool:
        if analysis.nose_ratio > 0.58:
            self.right_frames += 1
            self.feedback = "Right turn detected"
        else:
            self.right_frames = 0
            self.feedback = "Turn your head right"

        if analysis.nose_ratio < 0.42:
            self.wrong_frames += 1
        else:
            self.wrong_frames = 0

        if self.right_frames >= 5:
            self.success = True
        if self.wrong_frames >= 5:
            self.wrong_action = True
            self.feedback = "Wrong direction"

        return self.success


def create_detector(challenge: ChallengeType, baseline: BaselineMetrics) -> BaseChallengeDetector:
    if challenge == ChallengeType.BLINK:
        return BlinkDetector(baseline)
    if challenge == ChallengeType.TURN_LEFT:
        return TurnLeftDetector(baseline)
    if challenge == ChallengeType.TURN_RIGHT:
        return TurnRightDetector(baseline)
    if challenge == ChallengeType.OPEN_MOUTH:
        return OpenMouthDetector(baseline)
    raise ValueError(f"Unsupported challenge: {challenge}")


class LivenessSession:
    def __init__(self, *, max_retries: int = 0, challenge_timeout_sec: float = 8.0) -> None:
        self.max_retries = max_retries
        self.challenge_timeout_sec = challenge_timeout_sec
        self.reset()

    def reset(self, challenge_count: int = 2) -> None:
        self.challenge_count = max(1, min(challenge_count, 2))
        self.state = LivenessState.WAIT_CAMERA
        self.current_challenge: Optional[ChallengeType] = None
        self.pending_challenges: list[ChallengeType] = []
        self.completed_challenges: list[ChallengeType] = []
        self.retries_used = 0
        self.failure_reason: Optional[str] = None
        self.analysis: Optional[FrameAnalysis] = None
        self.baseline: Optional[BaselineMetrics] = None
        self.detector: Optional[BaseChallengeDetector] = None
        self.challenge_started_at: Optional[float] = None
        self.transition_until: Optional[float] = None
        self.info_message = "Allow camera access to begin"
        self.replacement_required = False

    def update(self, analysis: FrameAnalysis, now: Optional[float] = None) -> LivenessSnapshot:
        now = now or time.time()
        self.analysis = analysis

        if self.state in {LivenessState.SUCCESS, LivenessState.FAILED}:
            return self.snapshot()

        if self.state == LivenessState.CHALLENGE_ACTIVE:
            return self._update_active_challenge(analysis, now)

        if not analysis.face_detected:
            self.state = LivenessState.WAIT_FACE
            self.info_message = "Face not found"
            return self.snapshot()

        if not analysis.single_face:
            self.state = LivenessState.ALIGNING
            self.info_message = "Only one face is allowed"
            return self.snapshot()

        if analysis.stable_ok:
            self.state = LivenessState.READY
            self.baseline = BaselineMetrics(
                average_ear=analysis.average_ear,
                mouth_ratio=analysis.mouth_ratio,
                nose_ratio=analysis.nose_ratio,
            )
            self.info_message = "Alignment complete. Press start."
        else:
            self.state = LivenessState.ALIGNING
            self.info_message = analysis.guide_text

        return self.snapshot()

    def start_verification(self, challenge_count: Optional[int] = None) -> tuple[bool, str]:
        if challenge_count is not None:
            self.challenge_count = max(1, min(challenge_count, 2))

        if self.analysis is None or not self.analysis.stable_ok or self.baseline is None:
            return False, "얼굴 정렬이 완료된 뒤 시작할 수 있습니다."
        if self.state == LivenessState.CHALLENGE_ACTIVE:
            return False, "이미 인증이 진행 중입니다."

        available = list(ChallengeType)
        self.pending_challenges = random.sample(available, k=self.challenge_count)
        self.completed_challenges = []
        self.failure_reason = None
        self._activate_next_challenge(now=time.time())
        return True, "라이브니스 검증을 시작합니다."

    def snapshot(self) -> LivenessSnapshot:
        instruction_text = ""
        if self.current_challenge is not None:
            instruction_text = CHALLENGE_INSTRUCTIONS[self.current_challenge]
        elif self.state == LivenessState.SUCCESS:
            instruction_text = "Verification passed"
        elif self.state == LivenessState.FAILED:
            instruction_text = "Verification failed"

        guide_text = self.info_message
        if self.state == LivenessState.CHALLENGE_ACTIVE and self.detector is not None:
            guide_text = self.detector.feedback

        return LivenessSnapshot(
            state=self.state,
            guide_text=guide_text,
            instruction_text=instruction_text,
            current_challenge=self.current_challenge,
            pending_challenges=self.pending_challenges.copy(),
            completed_challenges=self.completed_challenges.copy(),
            retries_used=self.retries_used,
            max_retries=self.max_retries,
            failure_reason=self.failure_reason,
            ready_for_start=self.state == LivenessState.READY,
            replacement_required=self.replacement_required,
            analysis=self.analysis,
        )

    def _update_active_challenge(self, analysis: FrameAnalysis, now: float) -> LivenessSnapshot:
        if self.current_challenge is None and self.transition_until is not None:
            if now >= self.transition_until:
                self._activate_next_challenge(now)
            else:
                self.info_message = "Preparing next challenge"
                return self.snapshot()

        if self.detector is None or self.current_challenge is None:
            return self.snapshot()

        self.detector.update(analysis)

        if self.detector.wrong_action:
            self._mark_failed("동작 불일치")
            return self.snapshot()

        if self.detector.success:
            self.completed_challenges.append(self.current_challenge)
            self.current_challenge = None
            self.detector = None
            self.challenge_started_at = None

            if self.pending_challenges:
                self.transition_until = now + 1.0
                self.info_message = "Challenge complete. Get ready for the next one."
                return self.snapshot()

            self.state = LivenessState.SUCCESS
            self.info_message = "All challenges completed"
            return self.snapshot()

        if self.challenge_started_at is not None and (now - self.challenge_started_at) > self.challenge_timeout_sec:
            self._mark_failed(self._infer_failure_reason(analysis))

        return self.snapshot()

    def _activate_next_challenge(self, now: float) -> None:
        if not self.pending_challenges or self.baseline is None:
            self.state = LivenessState.SUCCESS
            self.info_message = "All challenges completed"
            return

        self.current_challenge = self.pending_challenges.pop(0)
        self.detector = create_detector(self.current_challenge, self.baseline)
        self.challenge_started_at = now
        self.transition_until = None
        self.state = LivenessState.CHALLENGE_ACTIVE
        self.info_message = CHALLENGE_INSTRUCTIONS[self.current_challenge]

    def _mark_failed(self, reason: str) -> None:
        self.failure_reason = reason
        self.retries_used += 1
        self.state = LivenessState.FAILED
        self.current_challenge = None
        self.pending_challenges = []
        self.detector = None
        self.challenge_started_at = None
        self.transition_until = None
        self.info_message = "Verification failed"

    def _infer_failure_reason(self, analysis: FrameAnalysis) -> str:
        if not analysis.face_detected:
            return "얼굴 없음"
        if not analysis.single_face:
            return "얼굴 여러 명"
        if not (analysis.center_ok and analysis.size_ok and analysis.front_facing):
            return "정렬 불량"
        return "지령 시간 초과"
