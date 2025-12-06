from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import base64
from deepface import DeepFace
# Import model emotion của bạn ở đây...

app = FastAPI()

@app.websocket("/ws/verify")
async def websocket_verify(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # 1. Decode Base64 to Image
            # 2. Run DeepFace.find()
            # 3. await websocket.send_json({"user": "Nguyen Van A", "status": "ok"})
            pass
    except WebSocketDisconnect:
        print("Client disconnected")

@app.websocket("/ws/emotion")
async def websocket_emotion(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # 1. Decode Base64
            # 2. Run Emotion Model predict
            # 3. await websocket.send_json({"emotion": "happy"})
            pass
    except WebSocketDisconnect:
        print("Client disconnected")