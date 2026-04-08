# 지령 성공 여부를 판정하는 규칙 모음
# face analyzer가 만든 숫자를 보고 "이 동작이 성공했는가"만 판단
from __future__ import annotations

from dataclasses import dataclass

from .types import ChallengeType, FrameAnalysis


@dataclass(slots=True)
class BaselineMetrics:
    average_ear: float
    mouth_ratio: float
    nose_ratio: float


class BaseChallengeDetector:
    instruction_text = ""

    def __init__(self, baseline: BaselineMetrics) -> None:
        self.baseline = baseline
        self.success = False
        self.wrong_action = False
        self.feedback = "Waiting for action"

    def update(self, analysis: FrameAnalysis) -> bool:
        raise NotImplementedError


class BlinkDetector(BaseChallengeDetector):
    instruction_text = "Blink once"

    def __init__(self, baseline: BaselineMetrics) -> None:
        super().__init__(baseline)
        self.closed_frames = 0
        self.closed_detected = False

    def update(self, analysis: FrameAnalysis) -> bool:
        avg_ear = analysis.average_ear
        close_threshold = min(0.20, self.baseline.average_ear * 0.72)
        reopen_threshold = max(0.23, self.baseline.average_ear * 0.88)

        if avg_ear > 0 and avg_ear < close_threshold:
            self.closed_frames += 1
            self.feedback = "Eyes closed"
        else:
            if self.closed_frames >= 2:
                self.closed_detected = True
            self.closed_frames = 0
            self.feedback = "Blink detection in progress"

        if self.closed_detected and avg_ear > reopen_threshold:
            self.success = True
            self.feedback = "Blink detected"

        return self.success


class OpenMouthDetector(BaseChallengeDetector):
    instruction_text = "Open your mouth"

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
    instruction_text = "Turn your head left"

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
    instruction_text = "Turn your head right"

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
