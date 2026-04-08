import speech_recognition as sr 
from google import genai
from google.genai import types

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os

google_client = genai.Client()

llm_model = google_client.chats.create(
    model="gemini-2.5-flash-lite",
    config=types.GenerateContentConfig(
        system_instruction="당신은 매우 불친절합니다. 반드시 30자 이내로만 답해주세요."
    ),
)

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# 기본 음성 확인
# voices = elevenlabs.voices.get_all()
# for v in voices.voices:
#     if v.category == "premade":
#         print(v.voice_id, v.name)
#         audio = elevenlabs.text_to_speech.convert(
#             text="채영아, 잠시만 괜찮을까?",
#             voice_id=v.voice_id,  # "George" - browse voices at elevenlabs.io/app/voice-library
#             model_id="eleven_flash_v2_5",
#             output_format="mp3_44100_128",
#             )
        
#         play(audio)
#         print("=" * 30)


# 1. 마이크 열기
# r은 Recognizer
r = sr.Recognizer()
while True:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)

        # STEP1: 마이크 입력 받기
        audio = r.listen(source)
        print("인식 중입니다......")
        
        # STEP2: 텍스트 변환 (Whisper)
        user_input = r.recognize_openai(audio) 
        print(f"[USER] {user_input}")

        ############################
        # 종료 조건
        ############################
        if user_input.strip() in ["그만", "종료", "꺼"]:
            break

        # STEP3: LLM 사용자의 입력에 답하기
        answer = llm_model.send_message(user_input).text
        print(f"[AI] {answer}")

        audio = elevenlabs.text_to_speech.convert(
            text=answer,
            voice_id="iP95p4xoKVk53GoZ742B",  # "George" - browse voices at elevenlabs.io/app/voice-library
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
            )
        
        play(audio)
