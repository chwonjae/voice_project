from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import av
import cv2

from .analyzer import FaceAnalyzer
from .liveness import LivenessSession
from .matcher import FaceMatcher
from .types import (
    CHALLENGE_LABELS_KO,
    LIVENESS_STATE_LABELS_KO,
    FaceAuthSnapshot,
    FaceAuthStage,
    LivenessState,
)


GUIDE_TEXT_KO = {
    "No face detected": "얼굴이 화면에 보이도록 카메라를 바라봐주세요.",
    "Keep only one face in frame": "한 사람만 화면 안에 들어오도록 해주세요.",
    "Move closer to the camera": "카메라에 조금 더 가까이 와주세요.",
    "Move slightly farther back": "카메라와 거리를 조금만 벌려주세요.",
    "Center your face": "얼굴을 화면 중앙에 맞춰주세요.",
    "Look straight ahead": "정면을 바라봐주세요.",
    "Hold still for alignment": "좋아요. 정면을 유지해주세요.",
    "Face aligned": "등록된 얼굴과 비교할 준비가 되었습니다.",
    "Alignment complete. Press start.": "라이브니스 검증을 시작합니다.",
    "Preparing next challenge": "다음 지령을 준비하고 있습니다.",
    "Challenge complete. Get ready for the next one.": "첫 번째 지령이 완료되었습니다. 다음 지령을 준비해주세요.",
    "All challenges completed": "모든 라이브니스 지령을 통과했습니다.",
    "Verification failed": "라이브니스 검증에 실패했습니다.",
    "Blink detection in progress": "눈 깜빡임을 확인하고 있습니다.",
    "Eyes closed": "눈 감김이 감지되었습니다.",
    "Blink detected": "눈 깜빡임이 감지되었습니다.",
    "Open your mouth wider": "입을 더 벌려주세요.",
    "Mouth opening detected": "입 벌리기가 감지되었습니다.",
    "Turn your head left": "고개를 왼쪽으로 돌려주세요.",
    "Turn your head right": "고개를 오른쪽으로 돌려주세요.",
    "Left turn detected": "왼쪽 회전이 감지되었습니다.",
    "Right turn detected": "오른쪽 회전이 감지되었습니다.",
    "Wrong direction": "반대 방향으로 움직였습니다.",
}

MATCH_STREAK_REQUIRED = 4
MISMATCH_STREAK_LIMIT = 6


def translate_message(message: str) -> str:
    return GUIDE_TEXT_KO.get(message, message)


class FaceAuthVideoProcessor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._analyzer = FaceAnalyzer(draw_mesh=False)
        self._matcher = FaceMatcher()
        self._liveness = LivenessSession(max_retries=0, challenge_timeout_sec=8.0)
        self._reference_path: Optional[str] = None
        self._challenge_count = 2
        self._match_streak = 0
        self._mismatch_streak = 0
        self._snapshot = FaceAuthSnapshot(
            stage=FaceAuthStage.MATCHING,
            guide_text="카메라 권한을 허용해주세요.",
            badge_text="얼굴 일치 확인",
        )

    def configure(self, reference_image_path: str, challenge_count: int = 2) -> tuple[bool, str]:
        with self._lock:
            normalized_path = str(Path(reference_image_path).resolve())
            self._challenge_count = max(1, min(challenge_count, 2))

            if self._reference_path == normalized_path:
                return True, "기준 얼굴이 준비되었습니다."

            ok, message = self._matcher.load_reference(normalized_path)
            if not ok:
                self._snapshot = FaceAuthSnapshot(
                    stage=FaceAuthStage.FAILED,
                    guide_text=message,
                    badge_text="기준 얼굴 오류",
                    result_message=message,
                )
                return False, message

            self._reference_path = normalized_path
            self._analyzer.reset()
            self._liveness.reset(challenge_count=self._challenge_count)
            self._match_streak = 0
            self._mismatch_streak = 0
            self._snapshot = FaceAuthSnapshot(
                stage=FaceAuthStage.MATCHING,
                guide_text="등록된 얼굴과 현재 얼굴을 확인합니다.",
                badge_text="얼굴 일치 확인",
            )
            return True, message

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = cv2.flip(frame_bgr, 1)
        analysis, annotated = self._analyzer.analyze(frame_bgr)

        with self._lock:
            if self._snapshot.stage in {FaceAuthStage.FAILED, FaceAuthStage.SUCCESS}:
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            if self._reference_path is None:
                self._snapshot = FaceAuthSnapshot(
                    stage=FaceAuthStage.FAILED,
                    guide_text="등록된 얼굴 기준 사진을 찾지 못했습니다.",
                    badge_text="기준 얼굴 오류",
                    result_message="등록된 얼굴 기준 사진을 찾지 못했습니다.",
                )
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            if self._snapshot.stage == FaceAuthStage.MATCHING:
                self._update_matching(analysis)
            elif self._snapshot.stage == FaceAuthStage.LIVENESS:
                self._update_liveness(analysis)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_snapshot(self) -> FaceAuthSnapshot:
        with self._lock:
            return self._snapshot

    def _update_matching(self, analysis) -> None:
        if not analysis.face_detected:
            self._match_streak = 0
            self._mismatch_streak = 0
            self._snapshot = FaceAuthSnapshot(
                stage=FaceAuthStage.MATCHING,
                guide_text=translate_message(analysis.guide_text),
                badge_text="얼굴 일치 확인",
            )
            return

        if not analysis.single_face or not analysis.center_ok or not analysis.size_ok or not analysis.front_facing:
            self._match_streak = 0
            self._mismatch_streak = 0
            self._snapshot = FaceAuthSnapshot(
                stage=FaceAuthStage.MATCHING,
                guide_text=translate_message(analysis.guide_text),
                badge_text="얼굴 위치 조정",
            )
            return

        if not analysis.stable_ok:
            self._match_streak = 0
            self._mismatch_streak = 0
            self._snapshot = FaceAuthSnapshot(
                stage=FaceAuthStage.MATCHING,
                guide_text=translate_message(analysis.guide_text),
                badge_text="얼굴 안정화",
            )
            return

        match_result = self._matcher.compare(analysis.face_signature)
        if match_result.matched:
            self._match_streak += 1
            self._mismatch_streak = 0
            if self._match_streak >= MATCH_STREAK_REQUIRED:
                self._liveness.reset(challenge_count=self._challenge_count)
                self._liveness.update(analysis)
                self._liveness.start_verification(challenge_count=self._challenge_count)
                liveness_snapshot = self._liveness.snapshot()
                self._snapshot = FaceAuthSnapshot(
                    stage=FaceAuthStage.LIVENESS,
                    guide_text="얼굴 일치가 확인되었습니다. 라이브니스 검증을 시작합니다.",
                    badge_text="라이브니스 검증",
                    instruction_text=self._describe_instruction(liveness_snapshot),
                    match_score=match_result.score,
                    match_threshold=match_result.threshold,
                    liveness_snapshot=liveness_snapshot,
                )
                return

            self._snapshot = FaceAuthSnapshot(
                stage=FaceAuthStage.MATCHING,
                guide_text="등록 얼굴과 일치 여부를 확인하고 있습니다.",
                badge_text="얼굴 일치 확인",
                match_score=match_result.score,
                match_threshold=match_result.threshold,
            )
            return

        self._match_streak = 0
        self._mismatch_streak += 1
        if self._mismatch_streak >= MISMATCH_STREAK_LIMIT:
            message = "등록된 얼굴과 현재 얼굴이 일치하지 않습니다."
            self._snapshot = FaceAuthSnapshot(
                stage=FaceAuthStage.FAILED,
                guide_text=message,
                badge_text="얼굴 불일치",
                result_message=message,
                match_score=match_result.score,
                match_threshold=match_result.threshold,
            )
            return

        self._snapshot = FaceAuthSnapshot(
            stage=FaceAuthStage.MATCHING,
            guide_text="등록된 얼굴과 다른 얼굴로 인식되었습니다. 정면으로 다시 맞춰주세요.",
            badge_text="얼굴 비교 중",
            match_score=match_result.score,
            match_threshold=match_result.threshold,
        )

    def _update_liveness(self, analysis) -> None:
        liveness_snapshot = self._liveness.update(analysis)

        if liveness_snapshot.state == LivenessState.SUCCESS:
            message = "얼굴 인증과 라이브니스 검증이 완료되었습니다."
            self._snapshot = FaceAuthSnapshot(
                stage=FaceAuthStage.SUCCESS,
                guide_text=message,
                badge_text="검증 완료",
                result_message=message,
                liveness_snapshot=liveness_snapshot,
            )
            return

        if liveness_snapshot.state == LivenessState.FAILED:
            failure_reason = liveness_snapshot.failure_reason or "라이브니스 검증 실패"
            message = f"라이브니스 검증에 실패했습니다. ({failure_reason})"
            self._snapshot = FaceAuthSnapshot(
                stage=FaceAuthStage.FAILED,
                guide_text=message,
                badge_text="검증 실패",
                result_message=message,
                liveness_snapshot=liveness_snapshot,
            )
            return

        self._snapshot = FaceAuthSnapshot(
            stage=FaceAuthStage.LIVENESS,
            guide_text=translate_message(liveness_snapshot.guide_text),
            badge_text=self._describe_liveness_state(liveness_snapshot.state),
            instruction_text=self._describe_instruction(liveness_snapshot),
            liveness_snapshot=liveness_snapshot,
        )

    def _describe_instruction(self, snapshot) -> str:
        if snapshot.current_challenge is not None:
            return CHALLENGE_LABELS_KO[snapshot.current_challenge]
        if snapshot.state == LivenessState.SUCCESS:
            return "라이브니스 검증 완료"
        return ""

    def _describe_liveness_state(self, state: LivenessState) -> str:
        return LIVENESS_STATE_LABELS_KO.get(state, "라이브니스 검증")
