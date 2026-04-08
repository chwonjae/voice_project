# 얼굴을 실제로 분석하는 엔진
# MediaPipe Face Mesh를 사용해서 랜드마크를 뽑는다.
from __future__ import annotations

from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

from .types import FrameAnalysis


class FaceAnalyzer:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_FACE = 234
    RIGHT_FACE = 454
    NOSE_TIP = 1
    UPPER_LIP = 13
    LOWER_LIP = 14

    def __init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._drawing = mp.solutions.drawing_utils
        self._stable_counter = 0

    def reset(self) -> None:
        self._stable_counter = 0

    def analyze(self, frame_bgr: np.ndarray) -> tuple[FrameAnalysis, np.ndarray]:
        annotated = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        height, width = annotated.shape[:2]
        faces = results.multi_face_landmarks or []

        if not faces:
            self._stable_counter = 0
            analysis = FrameAnalysis(
                face_detected=False,
                single_face=False,
                center_ok=False,
                size_ok=False,
                stable_ok=False,
                left_ear=0.0,
                right_ear=0.0,
                mouth_ratio=0.0,
                nose_ratio=0.0,
                guide_text="No face detected",
                face_count=0,
                front_facing=False,
                stable_frame_count=0,
                frame_width=width,
                frame_height=height,
            )
            self._draw_overlay(annotated, analysis)
            return analysis, annotated

        face_landmarks = faces[0]
        pts = np.array(
            [[landmark.x * width, landmark.y * height] for landmark in face_landmarks.landmark],
            dtype=np.float32,
        )
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)
        face_box = (
            max(x_min, 0),
            max(y_min, 0),
            min(x_max, width - 1),
            min(y_max, height - 1),
        )

        face_width = max(face_box[2] - face_box[0], 1)
        face_height = max(face_box[3] - face_box[1], 1)
        center_x = (face_box[0] + face_box[2]) / 2.0
        center_y = (face_box[1] + face_box[3]) / 2.0
        center_ok = (
            abs(center_x - (width / 2.0)) <= width * 0.12
            and abs(center_y - (height / 2.0)) <= height * 0.12
        )

        face_width_ratio = face_width / float(width)
        size_ok = 0.25 <= face_width_ratio <= 0.55

        left_ear = self._compute_ear(pts, self.LEFT_EYE)
        right_ear = self._compute_ear(pts, self.RIGHT_EYE)
        mouth_ratio = self._compute_distance(pts[self.UPPER_LIP], pts[self.LOWER_LIP]) / float(face_height)
        left_face_x = pts[self.LEFT_FACE][0]
        right_face_x = pts[self.RIGHT_FACE][0]
        nose_x = pts[self.NOSE_TIP][0]
        denom = max(right_face_x - left_face_x, 1.0)
        nose_ratio = float((nose_x - left_face_x) / denom)
        front_facing = 0.45 <= nose_ratio <= 0.55

        is_align_ok = len(faces) == 1 and center_ok and size_ok and front_facing
        if is_align_ok:
            self._stable_counter += 1
        else:
            self._stable_counter = 0

        stable_ok = self._stable_counter >= 15
        guide_text = self._resolve_guide_text(
            face_count=len(faces),
            size_ok=size_ok,
            face_width_ratio=face_width_ratio,
            center_ok=center_ok,
            front_facing=front_facing,
            stable_ok=stable_ok,
        )

        analysis = FrameAnalysis(
            face_detected=True,
            single_face=len(faces) == 1,
            center_ok=center_ok,
            size_ok=size_ok,
            stable_ok=stable_ok,
            left_ear=float(left_ear),
            right_ear=float(right_ear),
            mouth_ratio=float(mouth_ratio),
            nose_ratio=float(nose_ratio),
            guide_text=guide_text,
            face_count=len(faces),
            front_facing=front_facing,
            stable_frame_count=self._stable_counter,
            frame_width=width,
            frame_height=height,
            face_box=face_box,
        )

        self._drawing.draw_landmarks(
            annotated,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._drawing.DrawingSpec(
                color=(64, 170, 255),
                thickness=1,
                circle_radius=1,
            ),
        )
        self._draw_overlay(annotated, analysis)
        return analysis, annotated

    def _compute_ear(self, points: np.ndarray, indices: Iterable[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [points[index] for index in indices]
        vertical = self._compute_distance(p2, p6) + self._compute_distance(p3, p5)
        horizontal = max(self._compute_distance(p1, p4), 1e-6)
        return float(vertical / (2.0 * horizontal))

    @staticmethod
    def _compute_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        return float(np.linalg.norm(point_a - point_b))

    def _resolve_guide_text(
        self,
        *,
        face_count: int,
        size_ok: bool,
        face_width_ratio: float,
        center_ok: bool,
        front_facing: bool,
        stable_ok: bool,
    ) -> str:
        if face_count == 0:
            return "No face detected"
        if face_count > 1:
            return "Keep only one face in frame"
        if not size_ok:
            if face_width_ratio < 0.25:
                return "Move closer to the camera"
            return "Move slightly farther back"
        if not center_ok:
            return "Center your face"
        if not front_facing:
            return "Look straight ahead"
        if not stable_ok:
            return "Hold still for alignment"
        return "Face aligned"

    def _draw_overlay(self, frame: np.ndarray, analysis: FrameAnalysis) -> None:
        lines = [
            f"Guide: {analysis.guide_text}",
            f"Faces: {analysis.face_count}",
            f"EAR(L/R): {analysis.left_ear:.3f} / {analysis.right_ear:.3f}",
            f"Mouth ratio: {analysis.mouth_ratio:.3f}",
            f"Nose ratio: {analysis.nose_ratio:.3f}",
            f"Stable frames: {analysis.stable_frame_count}",
        ]

        if analysis.face_box:
            x1, y1, x2, y2 = analysis.face_box
            box_color = (80, 200, 120) if analysis.stable_ok else (80, 160, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        y = 30
        for line in lines:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (32, 32, 32), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 26
