# uvicorn main:app --port 포트번호 --reload
from fastapi import FastAPI, UploadFile, File
from typing import List

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


##################################################################################
# 챗봇 엔드포인트 만들기
##################################################################################
from pydantic import BaseModel
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
#---------------------------------------------------
# 챗봇 엔드포인트 만들기 (기본)
#---------------------------------------------------
class ChatRequest(BaseModel):
    message: str

def chatbot(user_message):
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "당신은 친절한 챗봇입니다."},
            {"role": "user", "content": user_message}
        ]
    )

    return response.output_text

@app.post("/chat")
async def chat(req: ChatRequest):
    response = chatbot(req.message)

    return {"text": response}

#---------------------------------------------------
# 챗봇 엔드포인트 만들기 (히스토리 반영)
#---------------------------------------------------
# 요청 데이터 
# [
#     {"role": "user", "content": ""},
#     {"role": "ai", "content": ""},
#     {"role": "user", "content": ""},
#     ...
# ]

class Message(BaseModel):
    role: str
    content: str

class ChatHistoryRequest(BaseModel):
    history: List[Message]

def chatbot2(chat_history):
    input_list = [{"role": "system", "content": "당신은 친절한 챗봇입니다."}]
    for chat in chat_history:
        if chat.role == "ai":
            role = "assistant"
        else:
            role = "user"

        input_list.append(
            {"role": role, "content": chat.content}
        )

    response = client.responses.create(
        model="gpt-5-nano",
        input=input_list
    )

    return response.output_text

@app.post("/chat_with_history")
async def chat2(req: ChatHistoryRequest):
    response = chatbot2(req.history)

    return {"text": response}

