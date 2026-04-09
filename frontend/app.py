import base64
import os
import time
from html import escape
from pathlib import Path
from typing import Optional

import cv2
import httpx
import mediapipe as mp
import numpy as np
import streamlit as st
from face_auth import FaceAuthStage, FaceAuthVideoProcessor
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="SUL Bank Demo",
    page_icon="💶",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Demo Constants
# ============================================================
SAMPLE_SCREEN_SPLASH = "splash"
SAMPLE_SCREEN_HOME = "home"
SAMPLE_SCREEN_RECIPIENT = "recipient"
SAMPLE_SCREEN_AMOUNT = "amount"
SAMPLE_FACE_REGISTRATION_DIR = Path(__file__).resolve().parent / "data" / "registered_faces"
SAMPLE_FACE_IMAGE_NAME = "primary_user_face.jpg"

SAMPLE_AVAILABLE_BANK_OPTIONS = [
    "신한은행",
    "국민은행",
    "우리은행",
    "하나은행",
    "농협은행",
    "카카오뱅크",
    "토스뱅크",
    "기업은행",
]
SAMPLE_VOICE_MODE_FREE = "free"
SAMPLE_VOICE_MODE_TURN = "turn"
SAMPLE_VOICE_FREE_DEMO_MESSAGES = [
    {
        "role": "assistant",
        "text": "안녕하세요. 송금 전에 통화 상황을 짧게 같이 점검할게요. 지금 누구에게 어떤 이유로 보내려는지 편하게 말씀해 주세요.",
    },
    {
        "role": "user",
        "text": "상대가 카드 배송이 잘못됐다면서, 오늘 안에 확인금처럼 먼저 보내 달라고 했어요.",
    },
    {
        "role": "assistant",
        "text": "배송 문제를 이유로 먼저 송금을 요구하면 주의가 필요해요. 링크 클릭이나 앱 설치를 같이 요청받았는지도 확인해 볼게요.",
    },
    {
        "role": "user",
        "text": "메신저로 온 링크를 눌러서 본인 확인을 하라고도 했어요.",
    },
    {
        "role": "assistant",
        "text": "현재 디자인에서는 이런 흐름이 위험 신호 카드로 묶여 보이도록 구성했어요. 실제 STT와 판정 API는 다음 단계에서 연결하면 됩니다.",
    },
]
SAMPLE_VOICE_TURN_DEMO_FLOW = [
    {
        "title": "1차 점검",
        "prompt": "먼저 사용자가 지금 누구에게 얼마를 보내려는지 음성으로 남기는 단계입니다.",
        "user_example": "아는 동생이라고 해서 50만원만 먼저 보내 달라는 상황이에요.",
        "assistant_reply": "첫 응답을 바탕으로 상대의 정체와 송금 이유를 더 묻는 음성이 생성됩니다.",
    },
    {
        "title": "2차 재질문",
        "prompt": "다음 턴에서는 기관 사칭이나 앱 설치 요구가 있었는지 다시 확인합니다.",
        "user_example": "검찰은 아니고, 원격으로 확인해야 한다면서 앱을 깔라고 했어요.",
        "assistant_reply": "두 번째 응답 뒤에는 위험 근거와 즉시 행동 가이드가 함께 정리됩니다.",
    },
    {
        "title": "3차 종료 안내",
        "prompt": "마지막 턴에서는 송금 보류, 공식 번호 재확인, 가족 연락 여부를 정리하는 마무리 음성이 이어집니다.",
        "user_example": "일단 송금은 멈추고 직접 다시 확인해 볼게요.",
        "assistant_reply": "턴제형의 마지막 화면은 완료 애니메이션과 함께 최종 안내 카드로 닫히도록 디자인합니다.",
    },
]


# ============================================================
# Global Style
# ============================================================
def inject_sample_global_styles() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Pretendard", "Noto Sans KR", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top, rgba(31,113,255,0.06), transparent 18%),
                linear-gradient(180deg, #f4f6fa 0%, #eef2f7 100%);
        }

        section.main > div {
            max-width: 460px;
            padding-top: 0.75rem;
            padding-bottom: 1.2rem;
        }

        div[data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f5f7fb 0%, #eff3f8 100%);
        }

        div[data-testid="stHeader"] {
            background: transparent;
        }

        div[data-testid="collapsedControl"] {
            display: none;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .sample-screen-fade {
            animation: sampleFadeUp 0.45s ease-out;
        }

        @keyframes sampleFadeUp {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0px);
            }
        }

        .sample-brand-text {
            letter-spacing: -0.02em;
            letter-spacing: -0.02em;
            color: #1452d9;
        }

        .sample-muted-caption {
            color: #7b8597;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .sample-section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #162033;
            margin-bottom: 10px;
        }

        .sample-bank-card-title {
            color: rgba(255,255,255,0.92);
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .sample-bank-card-account {
            color: rgba(255,255,255,0.82);
            font-size: 0.82rem;
            margin-bottom: 12px;
        }

        .sample-bank-card-balance {
            color: white;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .sample-card-label {
            font-size: 0.76rem;
            color: #7b8597;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .sample-card-title {
            font-size: 1rem;
            color: #111827;
            font-weight: 700;
            margin-top: 2px;
        }

        .sample-card-subtitle {
            font-size: 0.84rem;
            color: #6b7280;
            margin-top: 5px;
            line-height: 1.45;
        }

        .sample-home-icon-badge {
            width: 36px;
            height: 36px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f2f6ff;
            font-size: 1rem;
        }

        .sample-hero-question {
            font-size: 1.85rem;
            line-height: 1.18;
            font-weight: 800;
            color: #111827;
            letter-spacing: -0.03em;
            margin-bottom: 4px;
        }

        .sample-amount-value {
            text-align: center;
            font-size: 2.25rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.04em;
            margin-top: 12px;
            margin-bottom: 8px;
        }

        .sample-amount-help {
            text-align: center;
            font-size: 0.88rem;
            color: #6b7280;
        }

        .sample-summary-chip {
            padding: 10px 12px;
            background: #f3f6fb;
            border: 1px solid #e5ebf4;
            border-radius: 14px;
            font-size: 0.82rem;
            color: #475569;
            margin-bottom: 8px;
        }

        .sample-bottom-nav-wrap {
            margin-top: 18px;
            padding: 8px 6px 0 6px;
            border-top: 1px solid rgba(15, 23, 42, 0.06);
        }

        .sample-bottom-nav-item {
            text-align: center;
            padding: 6px 0;
        }

        .sample-bottom-nav-icon {
            font-size: 1.06rem;
            line-height: 1.1;
            margin-bottom: 4px;
        }

        .sample-bottom-nav-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: #8a94a6;
        }

        .sample-bottom-nav-label-active,
        .sample-bottom-nav-icon-active {
            color: #1d4ed8;
        }

        .sample-splash-wrap {
            min-height: 770px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            text-align: center;
        }

        .sample-splash-logo {
            width: 92px;
            height: 92px;
            border-radius: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.4rem;
            background: linear-gradient(145deg, #1d4ed8 0%, #2563eb 55%, #60a5fa 100%);
            color: white;
            box-shadow: 0 18px 40px rgba(37, 99, 235, 0.28);
            margin-bottom: 18px;
        }

        .sample-splash-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.04em;
            margin-bottom: 8px;
        }

        .sample-splash-subtitle {
            font-size: 0.95rem;
            line-height: 1.55;
            color: #64748b;
            max-width: 270px;
            margin: 0 auto 18px auto;
        }

        .sample-inline-note {
            margin-top: 10px;
            font-size: 0.78rem;
            color: #94a3b8;
            text-align: center;
        }

        .sample-ai-review-hero {
            position: relative;
            min-height: 200px;
            border-radius: 24px;
            padding: 20px 16px;
            background:
                radial-gradient(circle at top, rgba(96,165,250,0.28), transparent 42%),
                linear-gradient(180deg, #eff6ff 0%, #f8fbff 100%);
            border: 1px solid rgba(37, 99, 235, 0.12);
            overflow: hidden;
        }

        .sample-ai-review-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            color: #1d4ed8;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(37, 99, 235, 0.10);
        }

        .sample-ai-review-title {
            margin-top: 14px;
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.03em;
        }

        .sample-ai-review-subtitle {
            margin-top: 6px;
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .sample-ai-voice-stage {
            margin-top: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 12px;
        }

        .sample-ai-orb-wrap {
            position: relative;
            width: 104px;
            height: 104px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .sample-ai-orb-ring,
        .sample-ai-orb-ring-delay {
            position: absolute;
            width: 104px;
            height: 104px;
            border-radius: 999px;
            border: 1px solid rgba(37, 99, 235, 0.18);
            animation: sampleVoicePulse 2.2s ease-out infinite;
        }

        .sample-ai-orb-ring-delay {
            animation-delay: 1.1s;
        }

        .sample-ai-orb-core {
            width: 72px;
            height: 72px;
            border-radius: 999px;
            background: linear-gradient(145deg, #2563eb 0%, #60a5fa 100%);
            box-shadow: 0 18px 32px rgba(37, 99, 235, 0.26);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.45rem;
            font-weight: 800;
            animation: sampleVoiceGlow 1.6s ease-in-out infinite;
        }

        .sample-ai-eq {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 6px;
            height: 36px;
        }

        .sample-ai-eq-bar {
            width: 8px;
            border-radius: 999px;
            background: linear-gradient(180deg, #93c5fd 0%, #2563eb 100%);
            transform-origin: bottom;
            animation: sampleVoiceBar 1s ease-in-out infinite;
        }

        .sample-ai-eq-bar:nth-child(1) { height: 14px; animation-delay: 0s; }
        .sample-ai-eq-bar:nth-child(2) { height: 28px; animation-delay: 0.15s; }
        .sample-ai-eq-bar:nth-child(3) { height: 22px; animation-delay: 0.3s; }
        .sample-ai-eq-bar:nth-child(4) { height: 30px; animation-delay: 0.45s; }
        .sample-ai-eq-bar:nth-child(5) { height: 18px; animation-delay: 0.6s; }

        .sample-ai-voice-status {
            font-size: 0.84rem;
            font-weight: 700;
            color: #1d4ed8;
        }

        .sample-ai-transcript {
            margin-top: 14px;
            padding: 14px 15px;
            border-radius: 18px 18px 18px 8px;
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.05);
            color: #1e293b;
            font-size: 0.92rem;
            line-height: 1.58;
        }

        .sample-ai-summary-card {
            margin-top: 12px;
            padding: 14px 15px;
            border-radius: 18px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }

        .sample-ai-summary-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .sample-ai-summary-value {
            margin-top: 4px;
            color: #0f172a;
            font-size: 0.95rem;
            font-weight: 700;
        }

        @keyframes sampleVoicePulse {
            0% {
                transform: scale(0.82);
                opacity: 0;
            }
            35% {
                opacity: 0.55;
            }
            100% {
                transform: scale(1.32);
                opacity: 0;
            }
        }

        @keyframes sampleVoiceGlow {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.07);
            }
        }

        @keyframes sampleVoiceBar {
            0%, 100% {
                transform: scaleY(0.45);
                opacity: 0.62;
            }
            50% {
                transform: scaleY(1.05);
                opacity: 1;
            }
        }

        div[data-testid="stButton"] > button {
            border-radius: 14px;
            min-height: 48px;
            font-weight: 700;
            border: 1px solid rgba(15, 23, 42, 0.07);
            transition: all 0.18s ease;
            box-shadow: none;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            border-color: rgba(37, 99, 235, 0.18);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: 14px !important;
            min-height: 52px !important;
            border-color: #dbe3ef !important;
            background: #fbfcff !important;
        }

        .sample-hidden-label {
            margin-top: -8px;
        }

        .sample-status-flag {
            min-height: 48px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 800;
            border: 1px solid transparent;
            letter-spacing: -0.01em;
        }

        .sample-status-flag-registered {
            background: rgba(34, 197, 94, 0.10);
            border-color: rgba(34, 197, 94, 0.18);
            color: #15803d;
        }

        .sample-status-flag-unregistered {
            background: rgba(248, 113, 113, 0.10);
            border-color: rgba(248, 113, 113, 0.18);
            color: #b91c1c;
        }

        .sample-face-help-card {
            padding: 14px 15px;
            border-radius: 18px;
            background: #f8fbff;
            border: 1px solid #dbeafe;
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .sample-face-auth-shell {
            padding: 16px;
            border-radius: 22px;
            background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
            border: 1px solid rgba(37, 99, 235, 0.10);
        }

        .sample-face-auth-header {
            font-size: 1.2rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
        }

        .sample-face-auth-subtitle {
            margin-top: 6px;
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .sample-face-auth-step-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 16px;
        }

        .sample-face-auth-step {
            padding: 10px 12px;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            text-align: center;
            font-size: 0.8rem;
            font-weight: 700;
            color: #64748b;
        }

        .sample-face-auth-step-active {
            background: #eff6ff;
            border-color: rgba(37, 99, 235, 0.28);
            color: #1d4ed8;
        }

        .sample-face-auth-step-done {
            background: rgba(34, 197, 94, 0.10);
            border-color: rgba(34, 197, 94, 0.22);
            color: #15803d;
        }

        .sample-face-auth-card {
            margin-top: 14px;
            padding: 14px 15px;
            border-radius: 18px;
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .sample-face-auth-badge {
            display: inline-flex;
            align-items: center;
            padding: 7px 10px;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 0.75rem;
            font-weight: 800;
        }

        .sample-face-auth-guide {
            margin-top: 12px;
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.45;
        }

        .sample-face-auth-instruction {
            margin-top: 8px;
            color: #2563eb;
            font-size: 0.92rem;
            font-weight: 800;
        }

        .sample-face-auth-meta {
            margin-top: 10px;
            color: #64748b;
            font-size: 0.82rem;
        }

        .sample-face-auth-result {
            margin-top: 16px;
            min-height: 240px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            border-radius: 22px;
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
            border: 1px solid rgba(37, 99, 235, 0.10);
        }

        .sample-face-auth-result-success {
            background: linear-gradient(180deg, #effdf5 0%, #ffffff 100%);
        }

        .sample-face-auth-result-failed {
            background: linear-gradient(180deg, #fff5f5 0%, #ffffff 100%);
        }

        .sample-face-auth-result-ring,
        .sample-face-auth-result-ring-delay {
            position: absolute;
            width: 132px;
            height: 132px;
            border-radius: 999px;
            border: 1px solid rgba(37, 99, 235, 0.18);
            animation: sampleFaceAuthRing 2s ease-out infinite;
        }

        .sample-face-auth-result-ring-delay {
            animation-delay: 0.8s;
        }

        .sample-face-auth-result-success .sample-face-auth-result-ring,
        .sample-face-auth-result-success .sample-face-auth-result-ring-delay {
            border-color: rgba(34, 197, 94, 0.22);
        }

        .sample-face-auth-result-failed .sample-face-auth-result-ring,
        .sample-face-auth-result-failed .sample-face-auth-result-ring-delay {
            border-color: rgba(239, 68, 68, 0.22);
        }

        .sample-face-auth-result-icon {
            width: 82px;
            height: 82px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            font-weight: 800;
            color: white;
            z-index: 1;
        }

        .sample-face-auth-result-icon-success {
            background: linear-gradient(145deg, #16a34a 0%, #4ade80 100%);
            box-shadow: 0 16px 32px rgba(34, 197, 94, 0.24);
        }

        .sample-face-auth-result-icon-failed {
            background: linear-gradient(145deg, #dc2626 0%, #fb7185 100%);
            box-shadow: 0 16px 32px rgba(239, 68, 68, 0.22);
            animation: sampleFaceAuthShake 0.7s ease-in-out 1;
        }

        .sample-face-auth-result-title {
            margin-top: 16px;
            font-size: 1.24rem;
            font-weight: 800;
            color: #0f172a;
            z-index: 1;
        }

        .sample-face-auth-result-message {
            margin-top: 8px;
            max-width: 290px;
            text-align: center;
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.55;
            z-index: 1;
        }

        .sample-voice-shell {
            padding: 18px;
            border-radius: 24px;
            background:
                radial-gradient(circle at top, rgba(96, 165, 250, 0.16), transparent 42%),
                linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
            border: 1px solid rgba(37, 99, 235, 0.10);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }

        .sample-voice-title {
            font-size: 1.28rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.03em;
        }

        .sample-voice-subtitle {
            margin-top: 6px;
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .sample-voice-mode-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 16px;
        }

        .sample-voice-mode-card {
            min-height: 188px;
            padding: 16px;
            border-radius: 20px;
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .sample-voice-mode-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 28px rgba(37, 99, 235, 0.10);
        }

        .sample-voice-mode-icon {
            width: 46px;
            height: 46px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            background: #eff6ff;
            color: #1d4ed8;
        }

        .sample-voice-mode-title {
            margin-top: 14px;
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
        }

        .sample-voice-mode-body {
            margin-top: 6px;
            color: #64748b;
            font-size: 0.86rem;
            line-height: 1.55;
        }

        .sample-voice-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }

        .sample-voice-chip {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            color: #475569;
            font-size: 0.74rem;
            font-weight: 700;
        }

        .sample-voice-recorder-card {
            margin-top: 16px;
            padding: 18px 14px;
            border-radius: 20px;
            background: linear-gradient(180deg, #eff6ff 0%, #f8fbff 100%);
            border: 1px solid rgba(37, 99, 235, 0.12);
            text-align: center;
        }

        .sample-voice-recorder-orb {
            position: relative;
            width: 88px;
            height: 88px;
            margin: 0 auto 12px auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .sample-voice-recorder-ring,
        .sample-voice-recorder-ring-delay {
            position: absolute;
            width: 88px;
            height: 88px;
            border-radius: 999px;
            border: 1px solid rgba(37, 99, 235, 0.18);
            animation: sampleVoicePulse 2.2s ease-out infinite;
        }

        .sample-voice-recorder-ring-delay {
            animation-delay: 1s;
        }

        .sample-voice-recorder-core {
            width: 58px;
            height: 58px;
            border-radius: 999px;
            background: linear-gradient(145deg, #2563eb 0%, #60a5fa 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
            font-weight: 800;
            box-shadow: 0 14px 28px rgba(37, 99, 235, 0.24);
        }

        .sample-voice-recorder-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
        }

        .sample-voice-recorder-caption {
            margin-top: 6px;
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .sample-voice-bubble {
            margin-top: 10px;
            padding: 14px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            line-height: 1.58;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
            animation: sampleFadeUp 0.35s ease-out;
        }

        .sample-voice-bubble-user {
            margin-left: 40px;
            background: linear-gradient(145deg, #1452d9 0%, #2563eb 100%);
            color: white;
            border-bottom-right-radius: 8px;
        }

        .sample-voice-bubble-ai {
            margin-right: 40px;
            background: white;
            color: #1e293b;
            border: 1px solid rgba(15, 23, 42, 0.06);
            border-bottom-left-radius: 8px;
        }

        .sample-voice-bubble-label {
            display: block;
            margin-bottom: 6px;
            font-size: 0.74rem;
            font-weight: 800;
            opacity: 0.78;
        }

        .sample-voice-summary-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 14px;
        }

        .sample-voice-summary-card {
            padding: 14px;
            border-radius: 18px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }

        .sample-voice-summary-label {
            font-size: 0.72rem;
            font-weight: 800;
            color: #64748b;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .sample-voice-summary-value {
            margin-top: 6px;
            color: #0f172a;
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1.45;
        }

        .sample-voice-list-card {
            margin-top: 12px;
            padding: 14px 15px;
            border-radius: 18px;
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.06);
        }

        .sample-voice-list-title {
            font-size: 0.86rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 8px;
        }

        .sample-voice-list-item {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.55;
            margin-bottom: 6px;
        }

        .sample-turn-progress {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 16px;
        }

        .sample-turn-progress-item {
            padding: 11px 10px;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            text-align: center;
            font-size: 0.78rem;
            font-weight: 800;
            color: #64748b;
        }

        .sample-turn-progress-done {
            background: rgba(34, 197, 94, 0.10);
            border-color: rgba(34, 197, 94, 0.22);
            color: #15803d;
        }

        .sample-turn-progress-active {
            background: #eff6ff;
            border-color: rgba(37, 99, 235, 0.24);
            color: #1d4ed8;
        }

        .sample-turn-stage-card {
            margin-top: 14px;
            padding: 16px;
            border-radius: 20px;
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .sample-turn-stage-badge {
            display: inline-flex;
            align-items: center;
            padding: 7px 10px;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 0.74rem;
            font-weight: 800;
        }

        .sample-turn-stage-title {
            margin-top: 12px;
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
        }

        .sample-turn-stage-copy {
            margin-top: 6px;
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .sample-turn-audio-card {
            margin-top: 12px;
            padding: 14px;
            border-radius: 18px;
            background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
            border: 1px solid rgba(37, 99, 235, 0.10);
        }

        .sample-turn-audio-wave {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 6px;
            height: 34px;
            margin-top: 12px;
        }

        .sample-turn-audio-wave span {
            width: 8px;
            border-radius: 999px;
            background: linear-gradient(180deg, #93c5fd 0%, #2563eb 100%);
            animation: sampleVoiceBar 1s ease-in-out infinite;
            transform-origin: bottom;
        }

        .sample-turn-audio-wave span:nth-child(1) { height: 14px; animation-delay: 0s; }
        .sample-turn-audio-wave span:nth-child(2) { height: 28px; animation-delay: 0.15s; }
        .sample-turn-audio-wave span:nth-child(3) { height: 18px; animation-delay: 0.3s; }
        .sample-turn-audio-wave span:nth-child(4) { height: 30px; animation-delay: 0.45s; }
        .sample-turn-audio-wave span:nth-child(5) { height: 20px; animation-delay: 0.6s; }

        .sample-turn-complete-card {
            margin-top: 16px;
            min-height: 220px;
            border-radius: 22px;
            background: linear-gradient(180deg, #effdf5 0%, #ffffff 100%);
            border: 1px solid rgba(34, 197, 94, 0.18);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }

        .sample-turn-complete-card .sample-face-auth-result-ring,
        .sample-turn-complete-card .sample-face-auth-result-ring-delay {
            border-color: rgba(34, 197, 94, 0.24);
        }

        .sample-turn-audio-player-card {
            margin-top: 16px;
            padding: 16px;
            border-radius: 20px;
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
            border: 1px solid rgba(37, 99, 235, 0.12);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
        }

        .sample-turn-audio-player-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
        }

        .sample-turn-audio-player-caption {
            margin-top: 6px;
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .sample-turn-risk-card {
            margin-top: 14px;
            padding: 16px;
            border-radius: 20px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
        }

        .sample-turn-risk-card-green {
            background: linear-gradient(180deg, #effdf5 0%, #ffffff 100%);
            border-color: rgba(34, 197, 94, 0.16);
        }

        .sample-turn-risk-card-yellow {
            background: linear-gradient(180deg, #fffbeb 0%, #ffffff 100%);
            border-color: rgba(245, 158, 11, 0.18);
        }

        .sample-turn-risk-card-orange {
            background: linear-gradient(180deg, #fff7ed 0%, #ffffff 100%);
            border-color: rgba(249, 115, 22, 0.20);
        }

        .sample-turn-risk-card-red {
            background: linear-gradient(180deg, #fef2f2 0%, #ffffff 100%);
            border-color: rgba(239, 68, 68, 0.22);
        }

        .sample-turn-risk-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }

        .sample-turn-risk-title {
            font-size: 0.96rem;
            font-weight: 800;
            color: #0f172a;
        }

        .sample-turn-risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 800;
            color: white;
        }

        .sample-turn-risk-badge-green { background: #16a34a; }
        .sample-turn-risk-badge-yellow { background: #d97706; }
        .sample-turn-risk-badge-orange { background: #ea580c; }
        .sample-turn-risk-badge-red { background: #dc2626; }

        .sample-turn-risk-score {
            margin-top: 8px;
            color: #475569;
            font-size: 0.84rem;
        }

        .sample-turn-risk-list {
            margin-top: 12px;
            display: grid;
            gap: 8px;
        }

        .sample-turn-risk-item {
            padding: 10px 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(15, 23, 42, 0.06);
            color: #334155;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .sample-turn-loading-card {
            margin-top: 12px;
            padding: 14px 15px;
            border-radius: 18px;
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
            border: 1px solid rgba(37, 99, 235, 0.12);
        }

        .sample-turn-loading-title {
            font-size: 0.88rem;
            font-weight: 800;
            color: #1d4ed8;
        }

        .sample-turn-loading-copy {
            margin-top: 6px;
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        @keyframes sampleFaceAuthRing {
            0% {
                transform: scale(0.84);
                opacity: 0;
            }
            35% {
                opacity: 0.5;
            }
            100% {
                transform: scale(1.28);
                opacity: 0;
            }
        }

        @keyframes sampleFaceAuthShake {
            0%, 100% { transform: translateX(0); }
            20% { transform: translateX(-6px); }
            40% { transform: translateX(6px); }
            60% { transform: translateX(-4px); }
            80% { transform: translateX(4px); }
        }

        @media (max-width: 520px) {
            section.main > div {
                padding-left: 0.4rem;
                padding-right: 0.4rem;
            }

            .sample-splash-wrap {
                min-height: 86vh;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# State Initialization
# ============================================================
def sample_get_registered_face_path() -> Path:
    return SAMPLE_FACE_REGISTRATION_DIR / SAMPLE_FACE_IMAGE_NAME


def initialize_sample_mock_bank_data() -> None:
    if "sample_fake_account_name" not in st.session_state:
        st.session_state.sample_fake_account_name = "SUL 주거래 우대통장"
    if "sample_fake_account_number" not in st.session_state:
        st.session_state.sample_fake_account_number = "110-482-190284"
    if "sample_fake_account_balance" not in st.session_state:
        st.session_state.sample_fake_account_balance = 12850320
    if "sample_fake_card_name" not in st.session_state:
        st.session_state.sample_fake_card_name = "SUL Prime 카드"
    if "sample_fake_card_masked_number" not in st.session_state:
        st.session_state.sample_fake_card_masked_number = "1234 •••• •••• 9402"


def initialize_sample_app_state() -> None:
    initialize_sample_mock_bank_data()

    defaults = {
        "sample_current_screen": SAMPLE_SCREEN_SPLASH,
        "sample_selected_bank_name": SAMPLE_AVAILABLE_BANK_OPTIONS[0],
        "sample_recipient_account_number": "",
        "sample_transfer_amount": 0,
        "sample_transfer_amount_display": "0원",
        "sample_home_tab": "home",
        "sample_is_demo_mode": True,
        "sample_recent_action_message": "",
        "sample_has_initialized_demo_data": True,
        "sample_should_auto_advance_from_splash": True,
        "sample_splash_has_advanced": False,
        "sample_is_face_registration_dialog_open": False,
        "sample_is_face_registered": sample_get_registered_face_path().exists(),
        "sample_registered_face_path": "",
        "sample_face_registration_message": "",
        "sample_face_registration_feedback_type": "",
        "sample_is_face_auth_popup_open": False,
        "sample_face_auth_stage": FaceAuthStage.IDLE.value,
        "sample_face_auth_badge_text": "",
        "sample_face_auth_guide_text": "",
        "sample_face_auth_instruction_text": "",
        "sample_face_auth_result_message": "",
        "sample_face_auth_match_score": None,
        "sample_face_auth_match_threshold": None,
        "sample_face_auth_auto_close_at": 0.0,
        "sample_face_auth_transfer_pending": False,
        "sample_face_auth_stream_nonce": 0,
        "sample_is_voice_mode_selector_popup_open": False,
        "sample_is_voice_free_chat_popup_open": False,
        "sample_is_voice_turn_chat_popup_open": False,
        "sample_is_voice_safe_result_popup_open": False,
        "sample_is_voice_risk_result_popup_open": False,
        "sample_voice_selected_mode": "",
        "sample_voice_selector_notice": "",
        "sample_voice_free_demo_message_count": 3,
        "sample_voice_free_show_analysis": True,
        "sample_voice_free_status_text": "대화 대기",
        "sample_voice_turn_stage": "opening_ready",
        "sample_voice_turn_index": 0,
        "sample_voice_turn_hint_text": "사용자 음성을 먼저 받는 턴제 흐름을 시각적으로 확인하는 단계입니다.",
        "sample_voice_turn_status_text": "첫 질문을 준비하고 있습니다.",
        "sample_voice_turn_session_id": "",
        "sample_voice_turn_history": [],
        "sample_voice_turn_latest_audio_base64": "",
        "sample_voice_turn_latest_result": None,
        "sample_voice_turn_is_processing": False,
        "sample_voice_turn_terminal_state": "",
        "sample_voice_turn_error_message": "",
        "sample_voice_turn_result_auto_close_at": 0.0,
        "sample_voice_turn_recorder_nonce": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    sample_refresh_face_registration_state()


# ============================================================
# Formatters
# ============================================================
def format_sample_currency_krw(amount: int) -> str:
    return f"{amount:,}원"


def format_sample_amount_for_display() -> None:
    amount = int(st.session_state.sample_transfer_amount)
    st.session_state.sample_transfer_amount_display = format_sample_currency_krw(amount)


# ============================================================
# Navigation
# ============================================================
def sample_navigate_to_screen(target_screen: str) -> None:
    st.session_state.sample_current_screen = target_screen
    st.rerun()


def sample_go_home() -> None:
    st.session_state.sample_current_screen = SAMPLE_SCREEN_HOME
    st.rerun()


def sample_go_back_from_current_screen() -> None:
    current_screen = st.session_state.sample_current_screen
    if current_screen == SAMPLE_SCREEN_RECIPIENT:
        st.session_state.sample_current_screen = SAMPLE_SCREEN_HOME
    elif current_screen == SAMPLE_SCREEN_AMOUNT:
        st.session_state.sample_current_screen = SAMPLE_SCREEN_RECIPIENT
    else:
        st.session_state.sample_current_screen = SAMPLE_SCREEN_HOME
    st.rerun()


def sample_refresh_face_registration_state() -> None:
    registered_face_path = sample_get_registered_face_path()
    is_registered = registered_face_path.exists()
    st.session_state.sample_is_face_registered = is_registered
    st.session_state.sample_registered_face_path = str(registered_face_path) if is_registered else ""


def sample_set_face_registration_feedback(message: str, feedback_type: str) -> None:
    st.session_state.sample_face_registration_message = message
    st.session_state.sample_face_registration_feedback_type = feedback_type


def sample_clear_face_registration_camera_state() -> None:
    if "sample_face_camera_capture" in st.session_state:
        del st.session_state["sample_face_camera_capture"]


def sample_open_face_registration_dialog() -> None:
    sample_clear_face_registration_camera_state()
    sample_set_face_registration_feedback("", "")
    st.session_state.sample_is_face_registration_dialog_open = True
    st.rerun()


def sample_close_face_registration_dialog() -> None:
    sample_clear_face_registration_camera_state()
    st.session_state.sample_is_face_registration_dialog_open = False
    st.rerun()


def sample_decode_camera_image(captured_face) -> Optional[np.ndarray]:
    image_bytes = captured_face.getvalue()
    if not image_bytes:
        return None

    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)


def sample_validate_face_registration_image(image_bgr: Optional[np.ndarray]) -> tuple[bool, str]:
    if image_bgr is None:
        return False, "촬영한 이미지를 읽지 못했습니다. 다시 촬영해 주세요."

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    ) as face_detector:
        detection_result = face_detector.process(image_rgb)

    detections = detection_result.detections or []
    if len(detections) == 0:
        return False, "얼굴이 보이도록 다시 촬영해 주세요."
    if len(detections) > 1:
        return False, "한 사람만 화면에 나오도록 다시 촬영해 주세요."

    return True, "얼굴 사진이 등록되었습니다."


def sample_register_face_image(captured_face) -> tuple[bool, str]:
    image_bgr = sample_decode_camera_image(captured_face)
    is_valid, message = sample_validate_face_registration_image(image_bgr)
    if not is_valid:
        return False, message

    registered_face_path = sample_get_registered_face_path()
    registered_face_path.parent.mkdir(parents=True, exist_ok=True)
    registered_face_path.write_bytes(captured_face.getvalue())
    sample_refresh_face_registration_state()

    saved_path = registered_face_path.relative_to(Path(__file__).resolve().parent.parent).as_posix()
    return True, f"{message} 저장 위치: {saved_path}"


def sample_reset_face_auth_state() -> None:
    st.session_state.sample_face_auth_stage = FaceAuthStage.IDLE.value
    st.session_state.sample_face_auth_badge_text = ""
    st.session_state.sample_face_auth_guide_text = ""
    st.session_state.sample_face_auth_instruction_text = ""
    st.session_state.sample_face_auth_result_message = ""
    st.session_state.sample_face_auth_match_score = None
    st.session_state.sample_face_auth_match_threshold = None
    st.session_state.sample_face_auth_auto_close_at = 0.0
    st.session_state.sample_face_auth_transfer_pending = False


def sample_open_face_auth_popup() -> None:
    sample_reset_face_auth_state()
    st.session_state.sample_is_face_auth_popup_open = True
    st.session_state.sample_face_auth_stage = FaceAuthStage.MATCHING.value
    st.session_state.sample_face_auth_stream_nonce += 1


def sample_close_face_auth_popup() -> None:
    sample_reset_face_auth_state()
    st.session_state.sample_is_face_auth_popup_open = False


def sample_sync_face_auth_snapshot(snapshot) -> None:
    st.session_state.sample_face_auth_stage = snapshot.stage.value
    st.session_state.sample_face_auth_badge_text = snapshot.badge_text
    st.session_state.sample_face_auth_guide_text = snapshot.guide_text
    st.session_state.sample_face_auth_instruction_text = snapshot.instruction_text
    st.session_state.sample_face_auth_result_message = snapshot.result_message
    st.session_state.sample_face_auth_match_score = snapshot.match_score
    st.session_state.sample_face_auth_match_threshold = snapshot.match_threshold


def sample_mark_face_auth_terminal(success: bool, message: str) -> None:
    target_stage = FaceAuthStage.SUCCESS if success else FaceAuthStage.FAILED

    st.session_state.sample_face_auth_stage = target_stage.value
    st.session_state.sample_face_auth_badge_text = "검증 완료" if success else "검증 실패"
    st.session_state.sample_face_auth_guide_text = message
    st.session_state.sample_face_auth_instruction_text = ""
    st.session_state.sample_face_auth_result_message = message
    if float(st.session_state.sample_face_auth_auto_close_at or 0.0) <= 0.0:
        st.session_state.sample_face_auth_auto_close_at = time.time() + 1.8
    st.session_state.sample_face_auth_transfer_pending = success


def sample_complete_face_auth_flow_if_ready() -> None:
    if st.session_state.sample_face_auth_stage not in {FaceAuthStage.SUCCESS.value, FaceAuthStage.FAILED.value}:
        return
    if time.time() < float(st.session_state.sample_face_auth_auto_close_at or 0.0):
        return

    if st.session_state.sample_face_auth_transfer_pending:
        sample_close_face_auth_popup()
        sample_open_voice_mode_selector_popup()
        return
    else:
        st.session_state.sample_recent_action_message = "얼굴 인증이 완료되지 않아 송금이 취소되었습니다."

    sample_close_face_auth_popup()


def sample_reset_voice_chatbot_ui_state() -> None:
    st.session_state.sample_is_voice_mode_selector_popup_open = False
    st.session_state.sample_is_voice_free_chat_popup_open = False
    st.session_state.sample_is_voice_turn_chat_popup_open = False
    st.session_state.sample_is_voice_safe_result_popup_open = False
    st.session_state.sample_is_voice_risk_result_popup_open = False
    st.session_state.sample_voice_selected_mode = ""
    st.session_state.sample_voice_selector_notice = ""
    st.session_state.sample_voice_free_demo_message_count = 3
    st.session_state.sample_voice_free_show_analysis = True
    st.session_state.sample_voice_free_status_text = "대화 대기"
    st.session_state.sample_voice_turn_stage = "opening_ready"
    st.session_state.sample_voice_turn_index = 0
    st.session_state.sample_voice_turn_hint_text = "사용자 음성을 먼저 받는 턴제 흐름을 시각적으로 확인하는 단계입니다."
    st.session_state.sample_voice_turn_status_text = "첫 질문을 준비하고 있습니다."
    st.session_state.sample_voice_turn_session_id = ""
    st.session_state.sample_voice_turn_history = []
    st.session_state.sample_voice_turn_latest_audio_base64 = ""
    st.session_state.sample_voice_turn_latest_result = None
    st.session_state.sample_voice_turn_is_processing = False
    st.session_state.sample_voice_turn_terminal_state = ""
    st.session_state.sample_voice_turn_error_message = ""
    st.session_state.sample_voice_turn_result_auto_close_at = 0.0
    st.session_state.sample_voice_turn_recorder_nonce = 0


def sample_get_voice_backend_base_url() -> str:
    return os.getenv("VOICE_PHISHING_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def sample_raise_voice_backend_error(exc: Exception) -> RuntimeError:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        return RuntimeError(detail or "백엔드 요청에 실패했습니다.")
    if isinstance(exc, httpx.HTTPError):
        return RuntimeError("보이스피싱 백엔드와 연결하지 못했습니다.")
    return RuntimeError(str(exc))


def sample_create_turn_voice_session() -> dict:
    try:
        with httpx.Client(base_url=sample_get_voice_backend_base_url(), timeout=90.0) as client:
            response = client.post("/voice-phishing/turn/sessions")
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        raise sample_raise_voice_backend_error(exc) from exc


def sample_send_turn_voice_reply(session_id: str, audio_file) -> dict:
    audio_bytes = audio_file.getvalue()
    content_type = audio_file.type or "audio/wav"
    files = {
        "audio_file": (
            audio_file.name or f"turn_reply_{int(time.time())}.wav",
            audio_bytes,
            content_type,
        )
    }
    try:
        with httpx.Client(base_url=sample_get_voice_backend_base_url(), timeout=180.0) as client:
            response = client.post(f"/voice-phishing/turn/sessions/{session_id}/reply", files=files)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        raise sample_raise_voice_backend_error(exc) from exc


def sample_delete_turn_voice_session(silent: bool = True) -> None:
    session_id = str(st.session_state.sample_voice_turn_session_id or "")
    if not session_id:
        return

    st.session_state.sample_voice_turn_session_id = ""
    try:
        with httpx.Client(base_url=sample_get_voice_backend_base_url(), timeout=15.0) as client:
            client.delete(f"/voice-phishing/turn/sessions/{session_id}")
    except Exception:
        if not silent:
            raise


def sample_decode_voice_audio(audio_base64: str) -> bytes:
    if not audio_base64:
        return b""
    try:
        return base64.b64decode(audio_base64)
    except Exception:
        return b""


def sample_get_latest_turn_assistant_message() -> str:
    history = st.session_state.sample_voice_turn_history
    for item in reversed(history):
        if item.get("role") == "assistant":
            return str(item.get("text", ""))
    return ""


def sample_bootstrap_turn_voice_session() -> None:
    payload = sample_create_turn_voice_session()
    st.session_state.sample_voice_turn_session_id = payload["session_id"]
    st.session_state.sample_voice_turn_history = [
        {"role": "assistant", "text": payload["intro_message"]}
    ]
    st.session_state.sample_voice_turn_latest_audio_base64 = payload.get("intro_audio_base64", "")
    st.session_state.sample_voice_turn_latest_result = {
        "risk_level": payload.get("risk_level", "낮음"),
        "risk_score": payload.get("risk_score", 0),
        "suspected_types": payload.get("suspected_types", []),
        "key_evidence": payload.get("key_evidence", []),
        "immediate_action": payload.get("immediate_action", []),
        "conversation_status": payload.get("conversation_status", "in_progress"),
        "termination_reason": None,
        "system_message": payload.get("intro_message", ""),
    }
    st.session_state.sample_voice_turn_stage = "user_record_ready"
    st.session_state.sample_voice_turn_status_text = "질문을 들은 뒤 답변을 녹음해 주세요."
    st.session_state.sample_voice_turn_error_message = ""


def sample_transition_turn_voice_terminal_state(termination_reason: str | None) -> None:
    st.session_state.sample_is_voice_turn_chat_popup_open = False
    st.session_state.sample_voice_turn_terminal_state = termination_reason or ""
    if termination_reason == "safe_confirmed":
        st.session_state.sample_is_voice_safe_result_popup_open = True
        st.session_state.sample_voice_turn_result_auto_close_at = time.time() + 1.8
    else:
        st.session_state.sample_is_voice_risk_result_popup_open = True


def sample_apply_turn_voice_reply(payload: dict) -> None:
    history = list(st.session_state.sample_voice_turn_history)
    history.append({"role": "user", "text": payload["user_transcript"]})
    history.append({"role": "assistant", "text": payload["system_message"]})
    st.session_state.sample_voice_turn_history = history
    st.session_state.sample_voice_turn_latest_audio_base64 = payload.get("audio_base64", "")
    st.session_state.sample_voice_turn_latest_result = payload
    st.session_state.sample_voice_turn_error_message = ""
    st.session_state.sample_voice_turn_recorder_nonce += 1

    if payload.get("conversation_status") == "terminated":
        st.session_state.sample_voice_turn_status_text = "최종 결과를 정리하고 있습니다."
        sample_transition_turn_voice_terminal_state(payload.get("termination_reason"))
        return

    st.session_state.sample_voice_turn_stage = "user_record_ready"
    st.session_state.sample_voice_turn_status_text = "다음 질문이 준비됐어요. 답변을 녹음해 주세요."


def sample_open_voice_mode_selector_popup() -> None:
    sample_reset_voice_chatbot_ui_state()
    st.session_state.sample_is_voice_mode_selector_popup_open = True


def sample_open_voice_free_chat_popup() -> None:
    sample_reset_voice_chatbot_ui_state()
    st.session_state.sample_voice_selected_mode = SAMPLE_VOICE_MODE_FREE
    st.session_state.sample_is_voice_free_chat_popup_open = True
    st.session_state.sample_voice_free_status_text = "실시간 자유 대화 시안"


def sample_open_voice_turn_chat_popup() -> None:
    sample_reset_voice_chatbot_ui_state()
    st.session_state.sample_voice_selected_mode = SAMPLE_VOICE_MODE_TURN
    st.session_state.sample_is_voice_turn_chat_popup_open = True
    st.session_state.sample_voice_turn_stage = "booting"
    st.session_state.sample_voice_turn_status_text = "첫 질문을 준비하고 있습니다."
    st.session_state.sample_voice_turn_hint_text = "턴제 대화형 세션을 시작하는 중입니다."


def sample_close_voice_chatbot_ui(message: str = "", cleanup_session: bool = True) -> None:
    if cleanup_session:
        sample_delete_turn_voice_session(silent=True)
    sample_reset_voice_chatbot_ui_state()
    if message:
        st.session_state.sample_recent_action_message = message


def sample_finish_voice_chatbot_demo(mode_label: str) -> None:
    response = sample_submit_mock_transfer_request()
    sample_close_voice_chatbot_ui(
        f"{mode_label} UI 시안을 확인했습니다. {response['message']}"
    )


def sample_reveal_next_free_demo_message() -> None:
    next_count = min(
        st.session_state.sample_voice_free_demo_message_count + 1,
        len(SAMPLE_VOICE_FREE_DEMO_MESSAGES),
    )
    st.session_state.sample_voice_free_demo_message_count = next_count
    if next_count >= len(SAMPLE_VOICE_FREE_DEMO_MESSAGES):
        st.session_state.sample_voice_free_status_text = "위험 분석 카드까지 노출됨"
    else:
        st.session_state.sample_voice_free_status_text = "대화가 한 턴 더 확장됨"


def sample_advance_turn_chat_demo() -> None:
    current_stage = st.session_state.sample_voice_turn_stage
    current_index = int(st.session_state.sample_voice_turn_index)
    is_last_turn = current_index >= len(SAMPLE_VOICE_TURN_DEMO_FLOW) - 1

    if current_stage == "opening_ready":
        st.session_state.sample_voice_turn_stage = "user_record_ready"
        st.session_state.sample_voice_turn_hint_text = "사용자 녹음 카드가 열렸습니다."
    elif current_stage == "user_record_ready":
        st.session_state.sample_voice_turn_stage = "server_processing"
        st.session_state.sample_voice_turn_hint_text = "다음 단계 버튼으로 LLM 음성 생성 화면으로 넘어갑니다."
    elif current_stage == "server_processing":
        st.session_state.sample_voice_turn_stage = "ai_play_ready"
        st.session_state.sample_voice_turn_hint_text = "AI 음성 생성이 끝난 상태를 시각적으로 보여줍니다."
    elif current_stage == "ai_play_ready":
        if is_last_turn:
            st.session_state.sample_voice_turn_stage = "completed"
            st.session_state.sample_voice_turn_hint_text = "턴제 대화형의 완료 화면입니다."
        else:
            st.session_state.sample_voice_turn_stage = "next_ai_ready"
            st.session_state.sample_voice_turn_hint_text = "다음 질문으로 이어질 준비가 되었습니다."
    elif current_stage == "next_ai_ready":
        st.session_state.sample_voice_turn_index = current_index + 1
        st.session_state.sample_voice_turn_stage = "user_record_ready"
        st.session_state.sample_voice_turn_hint_text = "다음 사용자 녹음 턴이 열렸습니다."


def sample_get_turn_demo_primary_label() -> str:
    stage = st.session_state.sample_voice_turn_stage
    labels = {
        "opening_ready": "첫 녹음 단계 열기",
        "user_record_ready": "다음 스텝 보기",
        "server_processing": "생성된 음성 확인",
        "ai_play_ready": "재생 후 다음 턴으로 이동",
        "next_ai_ready": "다음 사용자 녹음 열기",
        "completed": "턴제형 UI 데모 종료",
    }
    return labels.get(stage, "다음 스텝 보기")


def sample_get_turn_demo_stage_badge() -> str:
    stage = st.session_state.sample_voice_turn_stage
    badges = {
        "opening_ready": "시작 안내",
        "user_record_ready": "사용자 녹음",
        "server_processing": "LLM 음성 생성",
        "ai_play_ready": "생성 음성 재생",
        "next_ai_ready": "다음 턴 대기",
        "completed": "완료 화면",
    }
    return badges.get(stage, "턴제 흐름")


def sample_render_voice_choice_card(icon: str, title: str, body: str, chips: list[str]) -> None:
    chip_markup = "".join([f'<span class="sample-voice-chip">{chip}</span>' for chip in chips])
    st.markdown(
        f"""
        <div class="sample-voice-mode-card">
            <div class="sample-voice-mode-icon">{icon}</div>
            <div class="sample-voice-mode-title">{title}</div>
            <div class="sample-voice-mode-body">{body}</div>
            <div class="sample-voice-chip-row">{chip_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_render_voice_bubble(role: str, text: str) -> None:
    bubble_class = "sample-voice-bubble sample-voice-bubble-user" if role == "user" else "sample-voice-bubble sample-voice-bubble-ai"
    label = "USER" if role == "user" else "AI REVIEW"
    safe_text = escape(text).replace("\n", "<br/>")
    st.markdown(
        f"""
        <div class="{bubble_class}">
            <span class="sample-voice-bubble-label">{label}</span>
            {safe_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_render_voice_recorder_card(title: str, caption: str, icon: str = "MIC") -> None:
    st.markdown(
        f"""
        <div class="sample-voice-recorder-card">
            <div class="sample-voice-recorder-orb">
                <div class="sample-voice-recorder-ring"></div>
                <div class="sample-voice-recorder-ring-delay"></div>
                <div class="sample-voice-recorder-core">{icon}</div>
            </div>
            <div class="sample-voice-recorder-title">{title}</div>
            <div class="sample-voice-recorder-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_get_risk_theme(latest_result: Optional[dict]) -> dict[str, str]:
    risk_score = int((latest_result or {}).get("risk_score", 0) or 0)
    if risk_score >= 60:
        return {"card": "sample-turn-risk-card-red", "badge": "sample-turn-risk-badge-red", "label": "고위험"}
    if risk_score >= 45:
        return {"card": "sample-turn-risk-card-orange", "badge": "sample-turn-risk-badge-orange", "label": "높음"}
    if risk_score >= 25:
        return {"card": "sample-turn-risk-card-yellow", "badge": "sample-turn-risk-badge-yellow", "label": "주의"}
    return {"card": "sample-turn-risk-card-green", "badge": "sample-turn-risk-badge-green", "label": "낮음"}


def sample_render_turn_audio_player(title: str, caption: str, audio_bytes: bytes, message_text: str) -> None:
    st.markdown(
        f"""
        <div class="sample-turn-audio-player-card">
            <div class="sample-turn-audio-player-title">{escape(title)}</div>
            <div class="sample-turn-audio-player-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.info("재생할 음성이 아직 준비되지 않았습니다.")
    if message_text:
        st.caption(message_text)


def sample_render_turn_risk_panel(latest_result: Optional[dict]) -> None:
    latest_result = latest_result or {}
    theme = sample_get_risk_theme(latest_result)
    risk_label = latest_result.get("risk_level", "낮음")
    risk_score = int(latest_result.get("risk_score", 0) or 0)
    suspected_types = latest_result.get("suspected_types", []) or []
    key_evidence = latest_result.get("key_evidence", []) or []

    suspected_items = "".join(
        [
            f'<div class="sample-turn-risk-item">{escape(item.get("type", ""))} · {int(item.get("score", 0) or 0)}점</div>'
            for item in suspected_types
        ]
    ) or '<div class="sample-turn-risk-item">아직 특정 수법이 좁혀지지 않았습니다.</div>'

    evidence_items = "".join(
        [f'<div class="sample-turn-risk-item">{escape(str(item))}</div>' for item in key_evidence]
    ) or '<div class="sample-turn-risk-item">첫 답변을 분석하면 핵심 근거가 여기에 표시됩니다.</div>'

    st.markdown(
        f"""
        <div class="sample-turn-risk-card {theme["card"]}">
            <div class="sample-turn-risk-header">
                <div class="sample-turn-risk-title">현재 위험도와 예상 수법</div>
                <div class="sample-turn-risk-badge {theme["badge"]}">{escape(str(risk_label))}</div>
            </div>
            <div class="sample-turn-risk-score">위험 점수 {risk_score}점 · 상태 {theme["label"]}</div>
            <div class="sample-turn-risk-list">{suspected_items}</div>
            <div class="sample-turn-risk-list">{evidence_items}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Sample Mock Functions (Future FastAPI Replacement Points)
# ============================================================
def sample_fetch_mock_account_summary() -> dict:
    return {
        "account_name": st.session_state.sample_fake_account_name,
        "account_number": st.session_state.sample_fake_account_number,
        "account_balance": st.session_state.sample_fake_account_balance,
    }


def sample_validate_mock_recipient_account(selected_bank_name: str, recipient_account_number: str) -> bool:
    if not selected_bank_name:
        return False
    sanitized = recipient_account_number.replace("-", "").replace(" ", "")
    return sanitized.isdigit() and len(sanitized) >= 8


def sample_submit_mock_transfer_request() -> dict:
    return {
        "success": True,
        "message": "데모용 송금 요청이 생성되었습니다. 실제 연동 시 FastAPI 요청으로 교체하세요.",
        "recipient_bank": st.session_state.sample_selected_bank_name,
        "recipient_account": st.session_state.sample_recipient_account_number,
        "amount": st.session_state.sample_transfer_amount,
    }

# ============================================================
# Components
# ============================================================
def render_sample_top_app_bar(
    left_button_label: Optional[str] = None,
    center_title: str = "",
    right_button_label: Optional[str] = None,
    left_action=None,
    right_action=None,
) -> None:
    left_col, center_col, right_col = st.columns([1, 4, 1])

    with left_col:
        if left_button_label:
            if st.button(left_button_label, key=f"top_left_{center_title}_{left_button_label}", use_container_width=True):
                if left_action:
                    left_action()
        else:
            st.write("")

    with center_col:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:10px; font-size:1rem; font-weight:800; color:#0f172a;">
                {center_title}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        if right_button_label:
            if st.button(right_button_label, key=f"top_right_{center_title}_{right_button_label}", use_container_width=True):
                if right_action:
                    right_action()
        else:
            st.write("")


def render_sample_home_header() -> None:
    left_col, center_col, right_col = st.columns([3.5, 1, 1])
    with left_col:
        st.markdown('<div><h2 class="sample-brand-text">💶 SUL Bank</h2></div>', unsafe_allow_html=True)
    with center_col:
        if st.button("📷", key="sample_home_camera_button", use_container_width=True):
            sample_open_face_registration_dialog()
    with right_col:
        is_registered = st.session_state.sample_is_face_registered
        flag_class_name = "sample-status-flag-registered" if is_registered else "sample-status-flag-unregistered"
        flag_label = "등록됨" if is_registered else "미등록"
        st.markdown(
            f'<div class="sample-status-flag {flag_class_name}">얼굴 {flag_label}</div>',
            unsafe_allow_html=True,
        )


def render_sample_face_registration_feedback() -> None:
    message = st.session_state.sample_face_registration_message
    if not message:
        return

    feedback_type = st.session_state.sample_face_registration_feedback_type
    if feedback_type == "success":
        st.success(message)
    elif feedback_type == "error":
        st.error(message)
    else:
        st.info(message)


def render_sample_section_header(title: str) -> None:
    st.markdown(f'<div class="sample-section-title">{title} &gt;</div>', unsafe_allow_html=True)


def render_sample_account_transfer_card() -> None:
    account_summary = sample_fetch_mock_account_summary()

    with stylable_container(
        key="sample_transfer_account_card_container",
        css_styles="""
            {
                background: linear-gradient(145deg, #1452d9 0%, #2563eb 58%, #4f8df7 100%);
                border-radius: 22px;
                padding: 18px 18px 18px 18px;
                box-shadow: 0 16px 34px rgba(37, 99, 235, 0.24);
                border: 1px solid rgba(255,255,255,0.12);
            }
            button {
                background: rgba(255,255,255,0.18) !important;
                color: white !important;
                border: 1px solid rgba(255,255,255,0.20) !important;
                border-radius: 14px !important;
                min-height: 46px !important;
            }
        """,
    ):
        top_left, top_right = st.columns([4, 1])
        with top_left:
            st.markdown(
                f"""
                <div class="sample-bank-card-title">💶 {account_summary['account_name']}</div>
                <div class="sample-bank-card-account">{account_summary['account_number']}</div>
                <div class="sample-bank-card-balance">{format_sample_currency_krw(account_summary['account_balance'])}</div>
                """,
                unsafe_allow_html=True,
            )
        with top_right:
            st.markdown(
                """
                <div style="display:flex; justify-content:flex-end; align-items:flex-start; height:100%;">
                    <div style="width:42px; height:42px; border-radius:14px; background:rgba(255,255,255,0.16); display:flex; align-items:center; justify-content:center; color:white; font-size:1.1rem; font-weight:700;">⇄</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        add_vertical_space(1)
        if st.button("이체", key="sample_go_transfer_from_account_card", use_container_width=True):
            sample_navigate_to_screen(SAMPLE_SCREEN_RECIPIENT)


def render_sample_mock_info_card(
    container_key: str,
    emoji: str,
    label: str,
    title: str,
    subtitle: str,
) -> None:
    with stylable_container(
        key=container_key,
        css_styles="""
            {
                background: rgba(255,255,255,0.94);
                border: 1px solid rgba(15, 23, 42, 0.05);
                border-radius: 20px;
                padding: 16px;
                padding-bottom: 48px;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        icon_col, text_col = st.columns([1, 4])
        with icon_col:
            st.markdown(f'<div class="sample-home-icon-badge">{emoji}</div>', unsafe_allow_html=True)
        with text_col:
            st.markdown(
                f"""
                <div class="sample-card-label">{label}</div>
                <div class="sample-card-title">{title}</div>
                <div class="sample-card-subtitle">{subtitle}</div>
                """,
                unsafe_allow_html=True,
            )


def render_sample_bottom_navigation() -> None:
    with stylable_container(
        key="sample_bottom_navigation_container",
        css_styles="""
            {
                background: rgba(255,255,255,0.72);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(15, 23, 42, 0.05);
                border-radius: 20px;
                padding: 8px 8px 6px 8px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            }
        """,
    ):
        st.markdown('<div class="sample-bottom-nav-wrap">', unsafe_allow_html=True)
        cols = st.columns(5)
        items = [
            ("🏠", "홈", True),
            ("💳", "금융", False),
            ("🎁", "혜택", False),
            ("📈", "주식", False),
            ("☰", "전체", False),
        ]
        for col, (icon, label, is_active) in zip(cols, items):
            with col:
                icon_class = "sample-bottom-nav-icon sample-bottom-nav-icon-active" if is_active else "sample-bottom-nav-icon"
                label_class = "sample-bottom-nav-label sample-bottom-nav-label-active" if is_active else "sample-bottom-nav-label"
                st.markdown(
                    f"""
                    <div class="sample-bottom-nav-item">
                        <div class="{icon_class}">{icon}</div>
                        <div class="{label_class}">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Screen Renderers
# ============================================================
def render_sample_splash_screen() -> None:
    st.markdown(
        """
        <div class="sample-splash-wrap">
            <div class="sample-splash-logo">💶</div>
            <div class="sample-splash-title">SUL Bank</div>
            <div class="sample-splash-subtitle">
                Streamlit로 구현한 모바일 뱅킹 시연 프로토타입<br/>
                secure, simple, seamless.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.sample_should_auto_advance_from_splash and not st.session_state.sample_splash_has_advanced:
        st.session_state.sample_splash_has_advanced = True
        time.sleep(1.2)
        sample_navigate_to_screen(SAMPLE_SCREEN_HOME)

    if st.button("데모 시작하기", key="sample_splash_start_button", use_container_width=True):
        sample_navigate_to_screen(SAMPLE_SCREEN_HOME)

    st.markdown('<div class="sample-inline-note">Demo UI • sample flow only</div>', unsafe_allow_html=True)


def render_sample_home_screen() -> None:
    render_sample_home_header()
    render_sample_face_registration_feedback()
    add_vertical_space(1)

    colored_header(
        label="내 금융 홈",
        description="실제 은행 앱처럼 보이는 송금 시연용 목업 화면입니다.",
        color_name="blue-70",
    )

    render_sample_section_header("은행")
    render_sample_account_transfer_card()

    add_vertical_space(1)
    render_sample_section_header("카드")
    render_sample_mock_info_card(
        container_key="sample_mock_card_section",
        emoji="💳",
        label="CARD",
        title=f"{st.session_state.sample_fake_card_name}",
        subtitle=f"{st.session_state.sample_fake_card_masked_number} · 이번 달 결제예정 428,500원",
    )

    add_vertical_space(1)
    render_sample_section_header("증권")
    render_sample_mock_info_card(
        container_key="sample_mock_stock_section",
        emoji="📈",
        label="INVEST",
        title="SUL 투자계좌",
        subtitle="해외주식 · ETF · 모의 수익률 +4.28%",
    )

    add_vertical_space(1)
    render_sample_section_header("보험")
    render_sample_mock_info_card(
        container_key="sample_mock_insurance_section",
        emoji="🛡️",
        label="INSURANCE",
        title="생활안심 보험 관리",
        subtitle="보장 확인, 납입 일정, 증명서 발급을 한 번에 확인하세요.",
    )

    render_sample_bottom_navigation()


def handle_sample_recipient_form_submission() -> None:
    is_valid = sample_validate_mock_recipient_account(
        selected_bank_name=st.session_state.sample_selected_bank_name,
        recipient_account_number=st.session_state.sample_recipient_account_number,
    )

    if not is_valid:
        st.session_state.sample_recent_action_message = "기관과 계좌번호를 정확히 입력해 주세요."
        return

    st.session_state.sample_recent_action_message = ""
    sample_navigate_to_screen(SAMPLE_SCREEN_AMOUNT)



def render_sample_recipient_screen() -> None:
    render_sample_top_app_bar(
        left_button_label="←",
        center_title="받는 분 선택",
        right_button_label="⌂",
        left_action=sample_go_back_from_current_screen,
        right_action=sample_go_home,
    )

    add_vertical_space(1)
    st.markdown('<div class="sample-hero-question">누구에게<br/>보낼까요?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-muted-caption">은행과 계좌번호를 입력하면 다음 단계로 이동합니다.</div>', unsafe_allow_html=True)
    add_vertical_space(1)

    with stylable_container(
        key="sample_recipient_form_container",
        css_styles="""
            {
                background: rgba(255,255,255,0.95);
                border-radius: 22px;
                padding: 16px;
                border: 1px solid rgba(15, 23, 42, 0.05);
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        st.selectbox(
            "기관 선택",
            options=SAMPLE_AVAILABLE_BANK_OPTIONS,
            key="sample_selected_bank_name",
            help="데모용 선택 필드입니다. 추후 실제 기관 조회 API와 연결할 수 있습니다.",
        )
        add_vertical_space(1)
        st.text_input(
            "계좌번호 입력",
            key="sample_recipient_account_number",
            placeholder="'-' 없이 계좌번호를 입력하세요",
            help="데모용 계좌번호 입력 필드입니다.",
        )

    if st.session_state.sample_recent_action_message:
        st.warning(st.session_state.sample_recent_action_message)

    add_vertical_space(2)
    if st.button("확인", key="sample_recipient_confirm_button", use_container_width=True):
        handle_sample_recipient_form_submission()


# ============================================================
# Amount Keypad Logic
# ============================================================
def handle_sample_amount_keypad_input(key: str) -> None:
    current_amount = str(st.session_state.sample_transfer_amount)

    if current_amount == "0":
        current_amount = ""

    next_amount_str = current_amount + key
    next_amount_str = next_amount_str.lstrip("0") or "0"

    try:
        st.session_state.sample_transfer_amount = int(next_amount_str)
    except ValueError:
        st.session_state.sample_transfer_amount = 0

    format_sample_amount_for_display()



def handle_sample_amount_delete() -> None:
    current_amount = str(st.session_state.sample_transfer_amount)
    next_amount_str = current_amount[:-1]
    if next_amount_str == "":
        next_amount_str = "0"
    st.session_state.sample_transfer_amount = int(next_amount_str)
    format_sample_amount_for_display()



def handle_sample_transfer_submit() -> None:
    if st.session_state.sample_transfer_amount <= 0:
        st.session_state.sample_recent_action_message = "보낼 금액을 입력해 주세요."
        return

    st.session_state.sample_recent_action_message = ""
    sample_open_face_auth_popup()



def render_sample_keypad_button(label: str, button_key: str, on_click_action=None, disabled: bool = False) -> None:
    if st.button(label, key=button_key, use_container_width=True, disabled=disabled):
        if on_click_action:
            on_click_action()
        st.rerun()


@st.dialog("얼굴 등록", width="large")
def render_sample_face_registration_dialog() -> None:
    render_sample_face_registration_feedback()
    st.markdown(
        """
        <div class="sample-face-help-card">
            카메라를 켜고 정면 얼굴을 촬영하면 프로젝트 내부에 실제 이미지가 저장됩니다.
            저장 전에 얼굴이 1명인지 확인하고, 얼굴이 없거나 여러 명이면 등록하지 않습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    registered_face_path = sample_get_registered_face_path()
    if st.session_state.sample_is_face_registered and registered_face_path.exists():
        add_vertical_space(1)
        st.image(str(registered_face_path), caption="현재 등록된 얼굴 사진", use_container_width=True)

    add_vertical_space(1)
    captured_face = st.camera_input(
        "본인 얼굴을 정면으로 촬영하세요.",
        key="sample_face_camera_capture",
    )

    if captured_face is None:
        st.info("촬영 후 등록 버튼을 누르면 사진이 저장됩니다.")
        if st.button("닫기", key="sample_face_registration_close_idle", use_container_width=True):
            sample_close_face_registration_dialog()
        return

    action_col_1, action_col_2 = st.columns(2)
    with action_col_1:
        if st.button("이 사진 등록", key="sample_face_registration_confirm", use_container_width=True):
            is_registered, message = sample_register_face_image(captured_face)
            feedback_type = "success" if is_registered else "error"
            sample_set_face_registration_feedback(message, feedback_type)
            if is_registered:
                st.session_state.sample_is_face_registration_dialog_open = False
            st.rerun()
    with action_col_2:
        if st.button("닫기", key="sample_face_registration_close_with_capture", use_container_width=True):
            sample_close_face_registration_dialog()


def render_sample_face_auth_steps() -> None:
    current_stage = st.session_state.sample_face_auth_stage
    matching_class = "sample-face-auth-step sample-face-auth-step-active"
    liveness_class = "sample-face-auth-step"
    result_class = "sample-face-auth-step"

    if current_stage in {FaceAuthStage.LIVENESS.value, FaceAuthStage.SUCCESS.value}:
        matching_class = "sample-face-auth-step sample-face-auth-step-done"
        liveness_class = "sample-face-auth-step sample-face-auth-step-active"
    if current_stage == FaceAuthStage.SUCCESS.value:
        liveness_class = "sample-face-auth-step sample-face-auth-step-done"
        result_class = "sample-face-auth-step sample-face-auth-step-done"
    if current_stage == FaceAuthStage.FAILED.value:
        result_class = "sample-face-auth-step sample-face-auth-step-active"

    st.markdown(
        f"""
        <div class="sample-face-auth-step-grid">
            <div class="{matching_class}">1. 얼굴 일치 확인</div>
            <div class="{liveness_class}">2. 라이브니스 검증</div>
            <div class="{result_class}">3. 결과 확인</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sample_face_auth_result_card(success: bool) -> None:
    result_class = "sample-face-auth-result-success" if success else "sample-face-auth-result-failed"
    icon_class = "sample-face-auth-result-icon-success" if success else "sample-face-auth-result-icon-failed"
    icon = "✓" if success else "!"
    title = "검증 완료" if success else "검증 실패"
    message = st.session_state.sample_face_auth_result_message or st.session_state.sample_face_auth_guide_text

    st.markdown(
        f"""
        <div class="sample-face-auth-result {result_class}">
            <div class="sample-face-auth-result-ring"></div>
            <div class="sample-face-auth-result-ring-delay"></div>
            <div class="sample-face-auth-result-icon {icon_class}">{icon}</div>
            <div class="sample-face-auth-result-title">{title}</div>
            <div class="sample-face-auth-result-message">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("송금 얼굴 인증", width="large", dismissible=False)
def render_sample_face_auth_popup() -> None:
    sample_complete_face_auth_flow_if_ready()
    if not st.session_state.sample_is_face_auth_popup_open:
        st.rerun()

    registered_face_path = sample_get_registered_face_path()
    if (
        not registered_face_path.exists()
        and st.session_state.sample_face_auth_stage not in {FaceAuthStage.SUCCESS.value, FaceAuthStage.FAILED.value}
    ):
        sample_mark_face_auth_terminal(False, "등록된 얼굴이 없어 송금을 진행할 수 없습니다.")

    current_stage = st.session_state.sample_face_auth_stage
    is_terminal = current_stage in {FaceAuthStage.SUCCESS.value, FaceAuthStage.FAILED.value}

    st.markdown('<div class="sample-face-auth-shell">', unsafe_allow_html=True)
    st.markdown('<div class="sample-face-auth-header">본인 확인을 진행할게요</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sample-face-auth-subtitle">등록된 얼굴과 현재 얼굴을 비교한 뒤, 2단계 라이브니스 검증으로 송금 전 본인 여부를 확인합니다.</div>',
        unsafe_allow_html=True,
    )
    render_sample_face_auth_steps()

    st.markdown(
        f"""
        <div class="sample-face-auth-card">
            <div class="sample-face-auth-badge">{st.session_state.sample_face_auth_badge_text or "얼굴 인증 준비"}</div>
            <div class="sample-face-auth-guide">{st.session_state.sample_face_auth_guide_text or "카메라를 준비하고 있습니다."}</div>
            <div class="sample-face-auth-instruction">{st.session_state.sample_face_auth_instruction_text}</div>
            <div class="sample-face-auth-meta">보낼 은행: {st.session_state.sample_selected_bank_name} · 계좌번호: {st.session_state.sample_recipient_account_number} · 금액: {st.session_state.sample_transfer_amount_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_terminal:
        render_sample_face_auth_result_card(current_stage == FaceAuthStage.SUCCESS.value)
        st.markdown("</div>", unsafe_allow_html=True)
        time.sleep(0.35)
        st.rerun()
        return

    add_vertical_space(1)
    stream_key = f"sample_face_auth_stream_{st.session_state.sample_face_auth_stream_nonce}"
    ctx = webrtc_streamer(
        key=stream_key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        desired_playing_state=True,
        video_processor_factory=FaceAuthVideoProcessor,
        async_processing=True,
        sendback_audio=False,
    )

    processor = ctx.video_processor if ctx else None
    if processor is not None:
        ok, message = processor.configure(str(registered_face_path), challenge_count=2)
        if not ok:
            sample_mark_face_auth_terminal(False, message)
        else:
            snapshot = processor.get_snapshot()
            sample_sync_face_auth_snapshot(snapshot)
            if snapshot.stage == FaceAuthStage.SUCCESS:
                sample_mark_face_auth_terminal(True, snapshot.result_message or snapshot.guide_text)
            elif snapshot.stage == FaceAuthStage.FAILED:
                sample_mark_face_auth_terminal(False, snapshot.result_message or snapshot.guide_text)

    match_score = st.session_state.sample_face_auth_match_score
    if match_score is not None:
        threshold = st.session_state.sample_face_auth_match_threshold
        threshold_text = f" / 기준 {threshold:.2f}" if threshold is not None else ""
        st.caption(f"얼굴 일치 점수 {match_score:.2f}{threshold_text}")

    if not ctx or not ctx.state.playing:
        st.info("브라우저에서 카메라 권한을 허용하면 얼굴 인증이 자동으로 시작됩니다.")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.sample_face_auth_stage in {FaceAuthStage.SUCCESS.value, FaceAuthStage.FAILED.value}:
        time.sleep(0.35)
        st.rerun()
        return

    if ctx and ctx.state.playing:
        time.sleep(0.35)
        st.rerun()
@st.dialog("보이스피싱 감지 방식 선택", width="large", dismissible=False)
def render_sample_voice_mode_selector_popup() -> None:
    st.markdown('<div class="sample-voice-shell">', unsafe_allow_html=True)
    st.markdown('<div class="sample-voice-title">송금 전 음성 확인 방식을 선택해 주세요</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sample-voice-subtitle">얼굴 인증이 끝나면 이 팝업에서 음성 확인 방식을 고릅니다. 이번 1차에서는 턴제 대화형만 실제 기능으로 연결하고, 자유 대화형은 다음 단계에서 이어집니다.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sample-voice-chip-row"><span class="sample-voice-chip">Mode Selector</span><span class="sample-voice-chip">Free Chat</span><span class="sample-voice-chip">Turn Based</span></div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2)
    with left_col:
        sample_render_voice_choice_card(
            icon="AI",
            title="자유 대화형",
            body="통화처럼 자연스럽게 이어지는 대화 로그와 위험 요약 카드가 함께 보이는 구조입니다. 이번 단계에서는 UI만 남겨 두고, 다음 구현에서 기능을 연결합니다.",
            chips=["실시간 로그", "자동 요약", "연속 재생"],
        )
        if st.button("자유 대화형 열기", key="sample_voice_selector_free", use_container_width=True):
            st.session_state.sample_voice_selector_notice = "자유 대화형은 다음 단계에서 연결할 예정입니다. 이번에는 턴제 대화형을 먼저 사용해 주세요."
            st.rerun()
    with right_col:
        sample_render_voice_choice_card(
            icon="STEP",
            title="턴제 대화형",
            body="사용자 음성 녹음, 다음 스텝 이동, AI 음성 생성과 재생을 단계별로 나눠 보여주는 버전입니다. 처음 보는 사용자에게 더 안전하고 명확한 흐름입니다.",
            chips=["순차 진행", "재생 버튼", "단계 강조"],
        )
        if st.button("턴제 대화형 열기", key="sample_voice_selector_turn", use_container_width=True):
            sample_open_voice_turn_chat_popup()
            st.rerun()

    if st.session_state.sample_voice_selector_notice:
        st.info(st.session_state.sample_voice_selector_notice)

    add_vertical_space(1)
    if st.button("이번에는 닫기", key="sample_voice_selector_close", use_container_width=True):
        sample_close_voice_chatbot_ui("보이스피싱 감지 챗봇 기능 선택 팝업을 닫았습니다.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


@st.dialog("자유 대화형 보이스피싱 감지", width="large", dismissible=False)
def render_sample_voice_free_chat_popup() -> None:
    visible_messages = SAMPLE_VOICE_FREE_DEMO_MESSAGES[: st.session_state.sample_voice_free_demo_message_count]

    st.markdown('<div class="sample-voice-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sample-ai-review-hero">
            <div class="sample-ai-review-chip">FREE TALK • Voice Review</div>
            <div class="sample-ai-review-title">자유롭게 이어지는 음성 대화 UX</div>
            <div class="sample-ai-review-subtitle">실제 STT, LLM, TTS가 붙으면 이 히어로 영역이 응답 생성 상태와 오디오 재생 애니메이션을 동시에 보여주는 중심 구간이 됩니다.</div>
            <div class="sample-ai-voice-stage">
                <div class="sample-ai-orb-wrap">
                    <div class="sample-ai-orb-ring"></div>
                    <div class="sample-ai-orb-ring-delay"></div>
                    <div class="sample-ai-orb-core">AI</div>
                </div>
                <div class="sample-ai-eq">
                    <div class="sample-ai-eq-bar"></div>
                    <div class="sample-ai-eq-bar"></div>
                    <div class="sample-ai-eq-bar"></div>
                    <div class="sample-ai-eq-bar"></div>
                    <div class="sample-ai-eq-bar"></div>
                </div>
                <div class="sample-ai-voice-status">현재 상태: 자유 대화형 시안</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_render_voice_recorder_card(
        title="음성 입력 자리 표시 영역",
        caption="실제 기능 연결 전이라 마이크 입력 대신, 음성 입력 위젯이 들어갈 자리와 애니메이션만 먼저 디자인했습니다.",
        icon="REC",
    )

    for message in visible_messages:
        sample_render_voice_bubble(message["role"], message["text"])

    left_summary, right_summary = st.columns(2)
    with left_summary:
        st.markdown(
            """
            <div class="sample-voice-summary-card">
                <div class="sample-voice-summary-label">Risk View</div>
                <div class="sample-voice-summary-value">주의 단계 요약 카드</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_summary:
        st.markdown(
            f"""
            <div class="sample-voice-summary-card">
                <div class="sample-voice-summary-label">Popup Status</div>
                <div class="sample-voice-summary-value">{st.session_state.sample_voice_free_status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.sample_voice_free_show_analysis:
        st.markdown(
            """
            <div class="sample-voice-list-card">
                <div class="sample-voice-list-title">핵심 근거 카드</div>
                <div class="sample-voice-list-item">• 상대가 먼저 송금을 요구하는 장면을 대화 직후에 바로 노출합니다.</div>
                <div class="sample-voice-list-item">• 링크 클릭, 앱 설치, 기관 사칭 여부를 하나의 분석 카드로 묶습니다.</div>
                <div class="sample-voice-list-item">• 실제 서버가 붙으면 턴마다 위험도와 근거 문구가 이 자리를 갱신합니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="sample-voice-list-card">
                <div class="sample-voice-list-title">즉시 행동 가이드</div>
                <div class="sample-voice-list-item">• 송금 보류와 공식 번호 재확인 버튼을 하단 CTA로 고정합니다.</div>
                <div class="sample-voice-list-item">• 보호자 연락이나 상담 연결 같은 보조 액션도 같은 레벨에서 보여줍니다.</div>
                <div class="sample-voice-list-item">• 실제 기능이 붙어도 디자인은 이 카드 구성을 유지할 수 있습니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    action_col_1, action_col_2, action_col_3 = st.columns(3)
    with action_col_1:
        if st.button("예시 대화 더 보기", key="sample_voice_free_more", use_container_width=True):
            sample_reveal_next_free_demo_message()
            st.rerun()
    with action_col_2:
        if st.button("분석 카드 토글", key="sample_voice_free_toggle", use_container_width=True):
            st.session_state.sample_voice_free_show_analysis = not st.session_state.sample_voice_free_show_analysis
            st.rerun()
    with action_col_3:
        if st.button("선택 팝업으로", key="sample_voice_free_back", use_container_width=True):
            sample_open_voice_mode_selector_popup()
            st.rerun()

    add_vertical_space(1)
    footer_col_1, footer_col_2 = st.columns(2)
    with footer_col_1:
        if st.button("자유 대화형 닫기", key="sample_voice_free_close", use_container_width=True):
            sample_close_voice_chatbot_ui("자유 대화형 보이스피싱 챗봇 UI 시안을 닫았습니다.")
            st.rerun()
    with footer_col_2:
        if st.button("이 흐름으로 송금 데모 마무리", key="sample_voice_free_finish", use_container_width=True):
            sample_finish_voice_chatbot_demo("자유 대화형 보이스피싱 챗봇")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


@st.dialog("턴제 대화형 보이스피싱 감지", width="large", dismissible=False)
def render_sample_voice_turn_chat_popup() -> None:
    if (
        not st.session_state.sample_voice_turn_session_id
        and not st.session_state.sample_voice_turn_error_message
    ):
        with st.spinner("첫 질문과 음성을 준비하고 있습니다..."):
            try:
                sample_bootstrap_turn_voice_session()
            except RuntimeError as exc:
                st.session_state.sample_voice_turn_stage = "error"
                st.session_state.sample_voice_turn_error_message = str(exc)
        st.rerun()
        return

    if not st.session_state.sample_voice_turn_session_id:
        st.markdown('<div class="sample-voice-shell">', unsafe_allow_html=True)
        st.markdown('<div class="sample-voice-title">턴제 대화형 보이스피싱 감지</div>', unsafe_allow_html=True)
        st.error(st.session_state.sample_voice_turn_error_message or "대화 세션을 시작하지 못했습니다.")
        retry_col, back_col = st.columns(2)
        with retry_col:
            if st.button("다시 시도", key="sample_voice_turn_retry_bootstrap", use_container_width=True):
                st.session_state.sample_voice_turn_error_message = ""
                st.session_state.sample_voice_turn_status_text = "첫 질문을 다시 준비하고 있습니다."
                st.rerun()
        with back_col:
            if st.button("선택 팝업으로", key="sample_voice_turn_bootstrap_back", use_container_width=True):
                sample_open_voice_mode_selector_popup()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    latest_result = st.session_state.sample_voice_turn_latest_result or {}
    latest_audio_bytes = sample_decode_voice_audio(
        st.session_state.sample_voice_turn_latest_audio_base64
    )
    latest_message = sample_get_latest_turn_assistant_message()
    recorder_key = f"sample_voice_turn_audio_input_{st.session_state.sample_voice_turn_recorder_nonce}"

    st.markdown('<div class="sample-voice-shell">', unsafe_allow_html=True)
    st.markdown('<div class="sample-voice-title">턴제 대화형 보이스피싱 감지</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sample-voice-subtitle">질문 음성을 먼저 듣고, 답변을 녹음한 뒤 전송하면 다음 질문과 위험도가 갱신됩니다.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sample-turn-loading-card">
            <div class="sample-turn-loading-title">현재 단계</div>
            <div class="sample-turn-loading-copy">{escape(st.session_state.sample_voice_turn_status_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.sample_voice_turn_error_message:
        st.error(st.session_state.sample_voice_turn_error_message)

    with stylable_container(
        key="sample_turn_audio_player_block",
        css_styles="""
            {
                margin-top: 8px;
            }
        """,
    ):
        sample_render_turn_audio_player(
            title="챗봇 질문 재생",
            caption="상단 플레이어에서 현재 질문을 재생할 수 있습니다.",
            audio_bytes=latest_audio_bytes,
            message_text=latest_message,
        )

    with stylable_container(
        key="sample_turn_audio_record_block",
        css_styles="""
            {
                margin-top: 8px;
                padding: 16px;
                border-radius: 20px;
                background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
                border: 1px solid rgba(37, 99, 235, 0.10);
                box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        st.markdown("**답변 녹음**")
        st.caption("녹음을 마친 뒤 `답변 보내기`를 누르면 다음 질문을 준비합니다.")
        recorded_audio = st.audio_input(
            "사용자 음성 녹음",
            key=recorder_key,
            disabled=st.session_state.sample_voice_turn_is_processing,
        )
        if st.button(
            "답변 보내기",
            key="sample_voice_turn_submit",
            use_container_width=True,
            disabled=st.session_state.sample_voice_turn_is_processing,
        ):
            if recorded_audio is None:
                st.warning("먼저 답변을 녹음해 주세요.")
            else:
                st.session_state.sample_voice_turn_is_processing = True
                st.session_state.sample_voice_turn_stage = "processing"
                st.session_state.sample_voice_turn_status_text = "답변을 분석하고 다음 질문을 준비하고 있습니다."
                try:
                    with st.status("응답을 준비하고 있어요...", expanded=True) as status_box:
                        status_box.write("음성을 업로드하고 있습니다.")
                        reply_payload = sample_send_turn_voice_reply(
                            st.session_state.sample_voice_turn_session_id,
                            recorded_audio,
                        )
                        status_box.write("STT와 위험도 분석을 정리하고 있습니다.")
                        status_box.write("챗봇 음성을 생성하고 있습니다.")
                        status_box.update(label="다음 질문 준비 완료", state="complete")
                    sample_apply_turn_voice_reply(reply_payload)
                except RuntimeError as exc:
                    st.session_state.sample_voice_turn_stage = "user_record_ready"
                    st.session_state.sample_voice_turn_status_text = "오류가 발생했습니다. 다시 녹음해 주세요."
                    st.session_state.sample_voice_turn_error_message = str(exc)
                finally:
                    st.session_state.sample_voice_turn_is_processing = False
                st.rerun()

    with stylable_container(
        key="sample_turn_history_block",
        css_styles="""
            {
                margin-top: 8px;
                padding: 16px;
                border-radius: 20px;
                background: rgba(255,255,255,0.96);
                border: 1px solid rgba(15, 23, 42, 0.06);
                box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        st.markdown("**대화 기록**")
        history = st.session_state.sample_voice_turn_history
        if history:
            for item in history:
                sample_render_voice_bubble(item["role"], item["text"])
        else:
            st.info("첫 질문이 준비되면 대화 기록이 표시됩니다.")

    sample_render_turn_risk_panel(latest_result)

    button_col_1, button_col_2 = st.columns(2)
    with button_col_1:
        if st.button("선택 팝업으로", key="sample_voice_turn_back", use_container_width=True):
            sample_close_voice_chatbot_ui(cleanup_session=True)
            sample_open_voice_mode_selector_popup()
            st.rerun()
    with button_col_2:
        if st.button("턴제형 닫기", key="sample_voice_turn_close", use_container_width=True):
            sample_close_voice_chatbot_ui("턴제 대화형 보이스피싱 감지를 닫았습니다.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


@st.dialog("송금 확인 완료", width="large", dismissible=False)
def render_sample_voice_safe_result_popup() -> None:
    latest_result = st.session_state.sample_voice_turn_latest_result or {}
    st.markdown('<div class="sample-voice-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sample-turn-complete-card">
            <div class="sample-face-auth-result-ring"></div>
            <div class="sample-face-auth-result-ring-delay"></div>
            <div class="sample-face-auth-result-icon sample-face-auth-result-icon-success">OK</div>
            <div class="sample-face-auth-result-title">안전하게 확인되었습니다</div>
            <div class="sample-face-auth-result-message">마지막 안내를 확인한 뒤 송금을 이어서 진행합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if latest_result.get("key_evidence"):
        st.markdown("**확인 근거**")
        for item in latest_result.get("key_evidence", []):
            st.markdown(f"- {item}")
    if latest_result.get("immediate_action"):
        st.markdown("**안내된 행동**")
        for item in latest_result.get("immediate_action", []):
            st.markdown(f"- {item}")

    if time.time() >= float(st.session_state.sample_voice_turn_result_auto_close_at or 0.0):
        response = sample_submit_mock_transfer_request()
        sample_close_voice_chatbot_ui(
            f"보이스피싱 위험이 낮아 송금을 이어서 진행했습니다. {response['message']}"
        )
        st.rerun()
        return

    st.caption("잠시 후 송금 완료 상태로 이동합니다.")
    time.sleep(0.35)
    st.rerun()


@st.dialog("위험 감지 결과", width="large", dismissible=False)
def render_sample_voice_risk_result_popup() -> None:
    latest_result = st.session_state.sample_voice_turn_latest_result or {}
    latest_audio_bytes = sample_decode_voice_audio(
        st.session_state.sample_voice_turn_latest_audio_base64
    )

    st.markdown('<div class="sample-voice-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sample-turn-complete-card" style="background: linear-gradient(180deg, #fff5f5 0%, #ffffff 100%); border-color: rgba(239, 68, 68, 0.18);">
            <div class="sample-face-auth-result-ring"></div>
            <div class="sample-face-auth-result-ring-delay"></div>
            <div class="sample-face-auth-result-icon sample-face-auth-result-icon-failed">!</div>
            <div class="sample-face-auth-result-title">위험 신호가 감지되었습니다</div>
            <div class="sample-face-auth-result-message">송금은 차단된 상태입니다. 마지막 안내 음성과 즉시 행동 가이드를 확인해 주세요.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_render_turn_audio_player(
        title="마지막 안내 음성",
        caption="종료 시점의 마지막 안내를 다시 들을 수 있습니다.",
        audio_bytes=latest_audio_bytes,
        message_text=str(latest_result.get("system_message", "")),
    )
    sample_render_turn_risk_panel(latest_result)

    with stylable_container(
        key="sample_turn_risk_action_block",
        css_styles="""
            {
                margin-top: 8px;
                padding: 16px;
                border-radius: 20px;
                background: rgba(255,255,255,0.96);
                border: 1px solid rgba(15, 23, 42, 0.06);
                box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        st.markdown("**즉시 행동 가이드**")
        immediate_actions = latest_result.get("immediate_action", []) or []
        if immediate_actions:
            for item in immediate_actions:
                st.markdown(f"- {item}")
        else:
            st.markdown("- 지금은 송금을 멈추고 공식 채널로 다시 확인해 주세요.")

    if st.button("송금 취소하고 돌아가기", key="sample_voice_risk_close", use_container_width=True):
        sample_close_voice_chatbot_ui("위험이 감지되어 이번 송금을 차단했습니다.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_sample_amount_screen() -> None:
    render_sample_top_app_bar(
        left_button_label="←",
        center_title="보낼 금액 입력",
        right_button_label="⌂",
        left_action=sample_go_back_from_current_screen,
        right_action=sample_go_home,
    )

    add_vertical_space(1)
    st.markdown('<div class="sample-hero-question">얼마를<br/>보낼까요?</div>', unsafe_allow_html=True)

    add_vertical_space(1)
    with stylable_container(
        key="sample_amount_summary_container",
        css_styles="""
            {
                background: rgba(255,255,255,0.95);
                border-radius: 22px;
                padding: 16px;
                border: 1px solid rgba(15, 23, 42, 0.05);
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        st.markdown(
            f'<div class="sample-summary-chip">받는 은행 · {st.session_state.sample_selected_bank_name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sample-summary-chip">계좌번호 · {st.session_state.sample_recipient_account_number}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sample-amount-value">{st.session_state.sample_transfer_amount_display}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sample-amount-help">숫자 키패드로 금액을 입력해 주세요.</div>', unsafe_allow_html=True)

    if st.session_state.sample_recent_action_message:
        st.info(st.session_state.sample_recent_action_message)

    add_vertical_space(1)
    with stylable_container(
        key="sample_transfer_submit_container",
        css_styles="""
            {
                background: transparent;
            }
            button {
                background: linear-gradient(145deg, #1452d9 0%, #2563eb 100%) !important;
                color: white !important;
                border: none !important;
                min-height: 52px !important;
                font-weight: 800 !important;
                box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
            }
        """,
    ):
        if st.button("전송", key="sample_amount_transfer_button", use_container_width=True):
            handle_sample_transfer_submit()
            st.rerun()

    add_vertical_space(1)
    keypad_rows = [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7", "8", "9"],
        ["00", "0", "←삭제"],
    ]

    for row_index, row_values in enumerate(keypad_rows):
        cols = st.columns(3)
        for col, key_value in zip(cols, row_values):
            with col:
                with stylable_container(
                    key=f"sample_keypad_container_{row_index}_{key_value}",
                    css_styles="""
                        {
                            background: transparent;
                        }
                        button {
                            background: rgba(255,255,255,0.96) !important;
                            color: #0f172a !important;
                            border: 1px solid rgba(15, 23, 42, 0.06) !important;
                            border-radius: 18px !important;
                            min-height: 66px !important;
                            font-size: 1.05rem !important;
                            font-weight: 800 !important;
                            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
                        }
                    """,
                ):
                    if key_value == "←삭제":
                        render_sample_keypad_button(
                            label="⌫",
                            button_key=f"sample_keypad_delete_{row_index}",
                            on_click_action=handle_sample_amount_delete,
                        )
                    else:
                        render_sample_keypad_button(
                            label=key_value,
                            button_key=f"sample_keypad_{row_index}_{key_value}",
                            on_click_action=lambda value=key_value: handle_sample_amount_keypad_input(value),
                        )

# ============================================================
# Router
# ============================================================
def render_sample_current_screen() -> None:
    current_screen = st.session_state.sample_current_screen

    if current_screen == SAMPLE_SCREEN_SPLASH:
        render_sample_splash_screen()
    elif current_screen == SAMPLE_SCREEN_HOME:
        render_sample_home_screen()
    elif current_screen == SAMPLE_SCREEN_RECIPIENT:
        render_sample_recipient_screen()
    elif current_screen == SAMPLE_SCREEN_AMOUNT:
        render_sample_amount_screen()
    else:
        render_sample_home_screen()


# ============================================================
# App Entrypoint
# ============================================================
def main() -> None:
    inject_sample_global_styles()
    initialize_sample_app_state()
    format_sample_amount_for_display()
    render_sample_current_screen()
    if st.session_state.sample_is_face_registration_dialog_open:
        render_sample_face_registration_dialog()
    if st.session_state.sample_is_face_auth_popup_open:
        render_sample_face_auth_popup()
    if st.session_state.sample_is_voice_mode_selector_popup_open:
        render_sample_voice_mode_selector_popup()
    if st.session_state.sample_is_voice_free_chat_popup_open:
        render_sample_voice_free_chat_popup()
    if st.session_state.sample_is_voice_turn_chat_popup_open:
        render_sample_voice_turn_chat_popup()
    if st.session_state.sample_is_voice_safe_result_popup_open:
        render_sample_voice_safe_result_popup()
    if st.session_state.sample_is_voice_risk_result_popup_open:
        render_sample_voice_risk_result_popup()


if __name__ == "__main__":
    main()
