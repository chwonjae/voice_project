# streamlit 화면
# 브라우저 카메라를 streamlit-webrtc로 받고, 내부의 LivenessVideoProcessor가 프레임마다 분석을 수행한다.
# 인증 시작, 세션 리셋, 1개/2개 지령 모드 선택도 여기서 처리
from __future__ import annotations

import threading
from typing import Optional

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from liveness.face_analyzer import FaceAnalyzer
from liveness.session_engine import LivenessSession
from liveness.types import CHALLENGE_LABELS_KO, SessionSnapshot, SessionState


STATE_LABELS_KO = {
    SessionState.WAIT_CAMERA: "카메라 권한 대기",
    SessionState.WAIT_FACE: "얼굴 인식 대기",
    SessionState.ALIGNING: "얼굴 정렬 중",
    SessionState.READY: "시작 가능",
    SessionState.CHALLENGE_ACTIVE: "지령 수행 중",
    SessionState.SUCCESS: "인증 통과",
    SessionState.FAILED: "인증 실패",
}

GUIDE_TEXT_KO = {
    "Allow camera access to begin": "카메라 권한을 허용해주세요.",
    "No face detected": "얼굴이 화면에 보이도록 카메라를 바라봐주세요.",
    "Keep only one face in frame": "한 사람만 화면 안에 들어오도록 해주세요.",
    "Move closer to the camera": "카메라에 조금 더 가까이 와주세요.",
    "Move slightly farther back": "카메라와 거리를 조금만 벌려주세요.",
    "Center your face": "얼굴을 화면 중앙에 맞춰주세요.",
    "Look straight ahead": "정면을 바라봐주세요.",
    "Hold still for alignment": "좋아요. 정면을 유지해주세요.",
    "Face aligned": "얼굴 정렬이 완료되었습니다.",
    "Face not found": "얼굴을 찾지 못했습니다.",
    "Only one face is allowed": "여러 얼굴이 감지되었습니다. 한 사람만 화면에 있어야 합니다.",
    "Alignment complete. Press start.": "얼굴 정렬이 완료되었습니다. 인증 시작 버튼을 눌러주세요.",
    "Preparing next challenge": "다음 지령을 준비하고 있습니다.",
    "Challenge complete. Get ready for the next one.": "첫 번째 지령이 완료되었습니다. 다음 지령을 준비해주세요.",
    "All challenges completed": "모든 지령을 통과했습니다.",
    "Verification passed": "라이브니스 검증을 통과했습니다.",
    "Verification failed": "라이브니스 검증에 실패했습니다.",
    "Verification failed. Reset and try again.": "인증에 실패했습니다. 세션을 초기화한 뒤 다시 시도해주세요.",
    "Retry limit exceeded. Use fallback verification.": "재시도 한도를 초과했습니다. 대체 인증을 진행해주세요.",
    "Waiting for action": "지령 수행을 기다리는 중입니다.",
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


class LivenessVideoProcessor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._analyzer = FaceAnalyzer()
        self._session = LivenessSession()
        self._snapshot = self._session.update_with_no_camera()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = cv2.flip(frame_bgr, 1)

        analysis, annotated = self._analyzer.analyze(frame_bgr)
        with self._lock:
            self._snapshot = self._session.update(analysis)
            self._draw_session_overlay(annotated, self._snapshot)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_snapshot(self) -> SessionSnapshot:
        with self._lock:
            return self._snapshot

    def reset(self, challenge_count: int) -> None:
        with self._lock:
            self._analyzer.reset()
            self._session.reset(challenge_count=challenge_count)
            self._snapshot = self._session.update_with_no_camera()

    def start_verification(self, challenge_count: int) -> tuple[bool, str]:
        with self._lock:
            ok, message = self._session.start_verification(challenge_count=challenge_count)
            self._snapshot = self._session.snapshot()
            return ok, message

    def _draw_session_overlay(self, frame, snapshot: SessionSnapshot) -> None:
        lines = [
            f"State: {snapshot.state.value}",
            f"Guide: {snapshot.guide_text}",
            f"Retries: {snapshot.retries_used}/{snapshot.max_retries}",
        ]

        if snapshot.current_challenge:
            lines.append(f"Challenge: {snapshot.current_challenge.value}")
        if snapshot.completed_challenges:
            completed = ", ".join(challenge.value for challenge in snapshot.completed_challenges)
            lines.append(f"Completed: {completed}")
        if snapshot.failure_reason:
            lines.append(f"Failure: {snapshot.failure_reason}")

        y = frame.shape[0] - (len(lines) * 24) - 10
        for line in lines:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 16, 16), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 235, 130), 1, cv2.LINE_AA)
            y += 24


def translate_message(message: str) -> str:
    return GUIDE_TEXT_KO.get(message, message)


def render_snapshot(snapshot: Optional[SessionSnapshot]) -> None:
    if snapshot is None:
        st.info("카메라 연결을 기다리는 중입니다.")
        return

    analysis = snapshot.analysis
    st.subheader(f"현재 상태: {STATE_LABELS_KO[snapshot.state]}")
    st.write(translate_message(snapshot.guide_text))

    if snapshot.current_challenge:
        st.success(f"현재 지령: {CHALLENGE_LABELS_KO[snapshot.current_challenge]}")
    elif snapshot.state == SessionState.READY:
        st.info("정렬이 완료되면 인증 시작 버튼을 눌러주세요.")

    if snapshot.completed_challenges:
        completed = [CHALLENGE_LABELS_KO[item] for item in snapshot.completed_challenges]
        st.write("완료한 지령:", ", ".join(completed))

    if snapshot.failure_reason:
        st.error(f"실패 사유: {snapshot.failure_reason}")
    if snapshot.replacement_required:
        st.warning("대체 인증 단계로 넘겨야 하는 상태입니다.")

    if analysis is None:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("EAR 평균", f"{analysis.average_ear:.3f}")
    col2.metric("입 비율", f"{analysis.mouth_ratio:.3f}")
    col3.metric("코 비율", f"{analysis.nose_ratio:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("얼굴 수", str(analysis.face_count))
    col5.metric("안정 프레임", str(analysis.stable_frame_count))
    col6.metric("재시도", f"{snapshot.retries_used}/{snapshot.max_retries}")


def main() -> None:
    st.set_page_config(page_title="Anchor Voice Liveness MVP", layout="wide")
    st.title("Anchor Voice 라이브니스 검증 MVP")
    st.caption("MediaPipe Face Mesh 기반 규칙형 챌린지-응답 인증 데모")

    challenge_count = st.sidebar.radio(
        "지령 개수",
        options=[1, 2],
        index=0,
        format_func=lambda value: f"{value}개 지령 모드",
    )
    st.sidebar.markdown(
        "\n".join(
            [
                "### 지원 지령",
                "- 눈 깜빡이기",
                "- 고개 왼쪽 돌리기",
                "- 고개 오른쪽 돌리기",
                "- 입 벌리기",
            ]
        )
    )
    st.sidebar.info("영상 오버레이를 보면서 얼굴 정렬을 맞춘 뒤 인증을 시작하세요.")

    ctx = webrtc_streamer(
        key="anchor-voice-liveness",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=LivenessVideoProcessor,
        async_processing=True,
    )

    processor = ctx.video_processor if ctx else None
    snapshot = processor.get_snapshot() if processor else None

    primary_label = "인증 시작"
    if snapshot is not None and snapshot.state == SessionState.FAILED and not snapshot.replacement_required:
        primary_label = "다시 시도"

    button_col1, button_col2 = st.columns(2)
    start_clicked = button_col1.button(primary_label, use_container_width=True, type="primary")
    reset_clicked = button_col2.button("세션 리셋", use_container_width=True)

    if reset_clicked and processor:
        processor.reset(challenge_count=challenge_count)
        snapshot = processor.get_snapshot()
        st.success("세션을 초기화했습니다.")

    if start_clicked:
        if processor is None:
            st.warning("먼저 카메라를 연결해주세요.")
        else:
            ok, message = processor.start_verification(challenge_count=challenge_count)
            snapshot = processor.get_snapshot()
            if ok:
                st.success(message)
            else:
                st.warning(message)

    if not ctx.state.playing:
        st.info("브라우저에서 카메라 권한을 허용한 뒤 Start 버튼으로 영상 전송을 시작해주세요.")

    render_snapshot(snapshot)

    st.markdown("---")
    st.markdown(
        "\n".join(
            [
                "### 수동 테스트 체크",
                "- 얼굴이 없을 때 `얼굴 인식 대기` 상태가 표시되는지 확인",
                "- 얼굴을 중앙에 맞추면 `시작 가능` 상태로 바뀌는지 확인",
                "- 1개 지령 모드에서 임의 지령 1개를 성공하면 인증 통과되는지 확인",
                "- 2개 지령 모드에서 첫 지령 성공 뒤 두 번째 지령이 이어지는지 확인",
                "- 정렬이 깨진 채 오래 머무르면 시간 초과로 실패하는지 확인",
            ]
        )
    )


if __name__ == "__main__":
    main()
