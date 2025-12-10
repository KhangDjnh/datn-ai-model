from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
from core.emotion_recognition import EmotionRecognizer
from core.face_verification import FaceVerifier
import logging

# Helper function để convert numpy types thành Python types
def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face AI System", description="API cho nhận diện cảm xúc và xác thực khuôn mặt")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo các model
try:
    emotion_recognizer = EmotionRecognizer(model_path='models/emotion_model_v2.hdf5')
    face_verifier = FaceVerifier(db_path="database")
    logger.info("✓ Các model được khởi tạo thành công")
except Exception as e:
    logger.error(f"✗ Lỗi khởi tạo model: {e}")

def decode_base64_image(base64_string):
    """
    Giải mã Base64 thành OpenCV image
    :param base64_string: Chuỗi Base64 của ảnh
    :return: Ảnh dạng numpy array (BGR)
    """
    try:
        # Loại bỏ header nếu có (data:image/jpeg;base64,)
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        # Giải mã Base64
        image_data = base64.b64decode(base64_string)
        
        # Chuyển thành numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode ảnh
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        logger.error(f"Lỗi giải mã Base64: {e}")
        return None

@app.websocket("/ws/verify")
async def websocket_verify(websocket: WebSocket):
    """
    WebSocket endpoint để xác thực khuôn mặt
    Client gửi Base64 image -> Server trả về thông tin nhận dạng
    """
    await websocket.accept()
    logger.info("Client kết nối đến /ws/verify")
    
    try:
        while True:
            # 1. Nhận dữ liệu Base64 từ client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON (nếu client gửi JSON object)
                payload = json.loads(data)
                image_base64 = payload.get("image", "")
            except:
                # Nếu không phải JSON, coi toàn bộ text là Base64
                image_base64 = data
            
            if not image_base64:
                await websocket.send_json({
                    "status": "error",
                    "message": "Không nhận được ảnh"
                })
                continue
            
            # 2. Giải mã Base64 thành ảnh
            image = decode_base64_image(image_base64)
            
            if image is None or image.size == 0:
                await websocket.send_json({
                    "status": "error",
                    "message": "Ảnh không hợp lệ"
                })
                continue
            
            # 3. Chạy Face Verification
            try:
                name, distance = face_verifier.verify(image)
                
                if name and distance is not None:
                    # Convert numpy types to Python types
                    name = str(name)
                    distance = float(distance)
                    is_match = bool(distance < 0.6)
                    
                    await websocket.send_json(convert_numpy_types({
                        "status": "ok",
                        "user": name,
                        "distance": distance,
                        "match": is_match,
                        "message": f"Tìm thấy: {name}" if is_match else f"Không khớp: {name} (distance: {distance:.4f})"
                    }))
                else:
                    await websocket.send_json({
                        "status": "not_found",
                        "message": "Không tìm thấy khuôn mặt phù hợp trong database"
                    })
            
            except Exception as e:
                logger.error(f"Lỗi verify: {e}")
                await websocket.send_json({
                    "status": "error",
                    "message": f"Lỗi xử lý: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("Client ngắt kết nối từ /ws/verify")

@app.websocket("/ws/emotion")
async def websocket_emotion(websocket: WebSocket):
    """
    WebSocket endpoint để nhận diện cảm xúc
    Client gửi Base64 image -> Server trả về cảm xúc được nhận diện
    """
    await websocket.accept()
    logger.info("Client kết nối đến /ws/emotion")
    
    try:
        while True:
            # 1. Nhận dữ liệu Base64 từ client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON (nếu client gửi JSON object)
                payload = json.loads(data)
                image_base64 = payload.get("image", "")
            except:
                # Nếu không phải JSON, coi toàn bộ text là Base64
                image_base64 = data
            
            if not image_base64:
                await websocket.send_json({
                    "status": "error",
                    "message": "Không nhận được ảnh"
                })
                continue
            
            # 2. Giải mã Base64 thành ảnh
            image = decode_base64_image(image_base64)
            
            if image is None or image.size == 0:
                await websocket.send_json({
                    "status": "error",
                    "message": "Ảnh không hợp lệ"
                })
                continue
            
            # 3. Chạy Emotion Recognition
            try:
                emotions = emotion_recognizer.predict(image)
                
                if emotions:
                    # Sắp xếp theo độ confident cao nhất
                    emotions_sorted = sorted(emotions, key=lambda x: x['confidence'], reverse=True)
                    
                    top_emotion = emotions_sorted[0]
                    
                    await websocket.send_json({
                        "status": "success",
                        "emotion": top_emotion['emotion'],
                        "confidence": float(top_emotion['confidence']),
                        "all_emotions": [
                            {"emotion": e['emotion'], "confidence": float(e['confidence'])}
                            for e in emotions_sorted
                        ],
                        "message": f"Cảm xúc chính: {top_emotion['emotion']} ({top_emotion['confidence']*100:.2f}%)"
                    })
                else:
                    await websocket.send_json({
                        "status": "no_face",
                        "message": "Không phát hiện được khuôn mặt trong ảnh"
                    })
            
            except Exception as e:
                logger.error(f"Lỗi emotion: {e}")
                await websocket.send_json({
                    "status": "error",
                    "message": f"Lỗi xử lý: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("Client ngắt kết nối từ /ws/emotion")

@app.get("/")
async def root():
    """Endpoint kiểm tra server đang chạy"""
    return {
        "status": "running",
        "message": "Face AI System API",
        "endpoints": {
            "verify": "ws://localhost:8000/ws/verify",
            "emotion": "ws://localhost:8000/ws/emotion"
        }
    }

@app.get("/health")
async def health_check():
    """Endpoint kiểm tra sức khỏe của server"""
    return {
        "status": "healthy",
        "emotion_model": emotion_recognizer.model is not None,
        "face_verifier": face_verifier is not None
    }