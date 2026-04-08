# 프로젝트의 상태머신
# face_analyzer와 challenge_rules를 묶어서 실제 인증 흐름으로 바꾸는 역할
# ex)
# - 얼굴이 없으면 WAIT_FACE
# - 얼굴 정렬이 되면 READY
# - 시작 버튼을 누르면 랜덤 지령 생성
# - 지령 수행 중이면 CHALLENGE_ACTIVE
# - 1개 또는 2개 지령이 모두 성공하면 SUCCESS
# - 시간 초과나 오동작이면 FAILED

from __future__ import annotations

import random
import time
from typing import Optional

from .challenge_rules import BaselineMetrics, BaseChallengeDetector, create_detector
from .types import CHALLENGE_INSTRUCTIONS, ChallengeType, FrameAnalysis, SessionSnapshot, SessionState


class LivenessSession:
    def __init__(self, *, max_retries: int = 2, challenge_timeout_sec: float = 10.0) -> None:
        self.max_retries = max_retries
        self.challenge_timeout_sec = challenge_timeout_sec
        self.reset()

    def reset(self, challenge_count: int = 1) -> None:
        self.challenge_count = max(1, min(challenge_count, 2))
        self.state = SessionState.WAIT_CAMERA
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

    def update_with_no_camera(self) -> SessionSnapshot:
        self.state = SessionState.WAIT_CAMERA
        self.info_message = "Allow camera access to begin"
        return self.snapshot()

    def update(self, analysis: FrameAnalysis, now: Optional[float] = None) -> SessionSnapshot:
        now = now or time.time()
        self.analysis = analysis

        if self.state in {SessionState.SUCCESS, SessionState.FAILED}:
            return self.snapshot()

        if self.state == SessionState.CHALLENGE_ACTIVE:
            return self._update_active_challenge(analysis, now)

        if not analysis.face_detected:
            self.state = SessionState.WAIT_FACE
            self.info_message = "Face not found"
            return self.snapshot()

        if not analysis.single_face:
            self.state = SessionState.ALIGNING
            self.info_message = "Only one face is allowed"
            return self.snapshot()

        if analysis.stable_ok:
            self.state = SessionState.READY
            self.baseline = BaselineMetrics(
                average_ear=analysis.average_ear,
                mouth_ratio=analysis.mouth_ratio,
                nose_ratio=analysis.nose_ratio,
            )
            self.info_message = "Alignment complete. Press start."
        else:
            self.state = SessionState.ALIGNING
            self.info_message = analysis.guide_text

        return self.snapshot()

    def start_verification(self, challenge_count: Optional[int] = None) -> tuple[bool, str]:
        if challenge_count is not None:
            self.challenge_count = max(1, min(challenge_count, 2))

        if self.analysis is None or not self.analysis.stable_ok or self.baseline is None:
            return False, "얼굴 정렬이 완료된 뒤 시작할 수 있습니다."
        if self.state == SessionState.CHALLENGE_ACTIVE:
            return False, "이미 인증이 진행 중입니다."
        if self.replacement_required:
            return False, "대체 인증이 필요한 상태입니다."

        available = list(ChallengeType)
        self.pending_challenges = random.sample(available, k=self.challenge_count)
        self.completed_challenges = []
        self.failure_reason = None
        self._activate_next_challenge(now=time.time())
        return True, "라이브니스 검증을 시작합니다."

    def snapshot(self) -> SessionSnapshot:
        current_instruction = ""
        if self.current_challenge is not None:
            current_instruction = CHALLENGE_INSTRUCTIONS[self.current_challenge]
        elif self.state == SessionState.SUCCESS:
            current_instruction = "Verification passed"
        elif self.state == SessionState.FAILED:
            current_instruction = "Verification failed"

        return SessionSnapshot(
            state=self.state,
            guide_text=(
                self.detector.feedback
                if self.state == SessionState.CHALLENGE_ACTIVE and self.detector is not None
                else self.info_message
            ),
            instruction_text=current_instruction,
            current_challenge=self.current_challenge,
            pending_challenges=self.pending_challenges.copy(),
            completed_challenges=self.completed_challenges.copy(),
            retries_used=self.retries_used,
            max_retries=self.max_retries,
            failure_reason=self.failure_reason,
            ready_for_start=self.state == SessionState.READY and not self.replacement_required,
            replacement_required=self.replacement_required,
            analysis=self.analysis,
        )

    def _update_active_challenge(self, analysis: FrameAnalysis, now: float) -> SessionSnapshot:
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

            self.state = SessionState.SUCCESS
            self.info_message = "All challenges completed"
            return self.snapshot()

        if self.challenge_started_at is not None and (now - self.challenge_started_at) > self.challenge_timeout_sec:
            self._mark_failed(self._infer_failure_reason(analysis))

        return self.snapshot()

    def _activate_next_challenge(self, now: float) -> None:
        if not self.pending_challenges or self.baseline is None:
            self.state = SessionState.SUCCESS
            self.info_message = "All challenges completed"
            return

        self.current_challenge = self.pending_challenges.pop(0)
        self.detector = create_detector(self.current_challenge, self.baseline)
        self.challenge_started_at = now
        self.transition_until = None
        self.state = SessionState.CHALLENGE_ACTIVE
        self.info_message = CHALLENGE_INSTRUCTIONS[self.current_challenge]

    def _mark_failed(self, reason: str) -> None:
        self.failure_reason = reason
        self.retries_used += 1
        self.state = SessionState.FAILED
        self.current_challenge = None
        self.pending_challenges = []
        self.detector = None
        self.challenge_started_at = None
        self.transition_until = None

        if self.retries_used > self.max_retries:
            self.replacement_required = True
            self.info_message = "Retry limit exceeded. Use fallback verification."
            return

        self.info_message = "Verification failed. Reset and try again."

    def _infer_failure_reason(self, analysis: FrameAnalysis) -> str:
        if not analysis.face_detected:
            return "얼굴 없음"
        if not analysis.single_face:
            return "얼굴 여러 명"
        if not (analysis.center_ok and analysis.size_ok and analysis.front_facing):
            return "정렬 불량"
        return "지령 시간 초과"
