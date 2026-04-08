from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .analyzer import FaceAnalyzer
from .types import FaceMatchResult


class FaceMatcher:
    def __init__(self, *, threshold: float = 0.92) -> None:
        self.threshold = threshold
        self._reference_signature: Optional[np.ndarray] = None
        self._reference_path: Optional[str] = None
        self._reference_analyzer = FaceAnalyzer(static_image_mode=True, draw_mesh=False)

    def load_reference(self, image_path: str) -> tuple[bool, str]:
        normalized_path = str(Path(image_path).resolve())
        if self._reference_signature is not None and self._reference_path == normalized_path:
            return True, "기준 얼굴이 준비되었습니다."

        image_bgr = cv2.imread(normalized_path)
        if image_bgr is None:
            return False, "등록된 얼굴 사진을 불러오지 못했습니다."

        analysis, _ = self._reference_analyzer.analyze(image_bgr)
        if not analysis.face_detected or analysis.face_signature is None:
            return False, "등록된 사진에서 얼굴을 찾지 못했습니다."
        if not analysis.single_face:
            return False, "등록 사진에는 한 사람의 얼굴만 있어야 합니다."

        self._reference_signature = analysis.face_signature.astype(np.float32)
        self._reference_path = normalized_path
        return True, "기준 얼굴이 준비되었습니다."

    def compare(self, live_signature: Optional[np.ndarray]) -> FaceMatchResult:
        if self._reference_signature is None or live_signature is None:
            return FaceMatchResult(matched=False, score=0.0, threshold=self.threshold)

        reference_vector = self._normalize_signature(self._reference_signature)
        live_vector = self._normalize_signature(live_signature.astype(np.float32))
        denominator = max(np.linalg.norm(reference_vector) * np.linalg.norm(live_vector), 1e-6)
        similarity = float(np.dot(reference_vector, live_vector) / denominator)
        return FaceMatchResult(
            matched=similarity >= self.threshold,
            score=similarity,
            threshold=self.threshold,
        )

    @staticmethod
    def _normalize_signature(signature: np.ndarray) -> np.ndarray:
        return (signature - signature.mean()) / (signature.std() + 1e-6)
