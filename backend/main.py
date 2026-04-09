from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.voice_phishing_service import (
    TurnReplyResponse,
    TurnSessionStartResponse,
    voice_phishing_service,
)


app = FastAPI(title="Voice Phishing Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/voice-phishing/turn/sessions", response_model=TurnSessionStartResponse)
async def create_turn_session() -> TurnSessionStartResponse:
    return voice_phishing_service.create_session()


@app.post(
    "/voice-phishing/turn/sessions/{session_id}/reply",
    response_model=TurnReplyResponse,
)
async def reply_turn_session(
    session_id: str,
    audio_file: UploadFile = File(...),
) -> TurnReplyResponse:
    audio_bytes = await audio_file.read()
    return voice_phishing_service.reply_to_turn(
        session_id=session_id,
        audio_bytes=audio_bytes,
        content_type=audio_file.content_type,
    )


@app.delete("/voice-phishing/turn/sessions/{session_id}")
async def delete_turn_session(session_id: str) -> dict[str, bool]:
    voice_phishing_service.delete_session(session_id)
    return {"deleted": True}
