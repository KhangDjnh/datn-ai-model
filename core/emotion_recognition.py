# core/emotion_recognition.py
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
from collections import deque

class EmotionRecognizer:
    def __init__(self, model_path='models/emotion_model_v2.hdf5', classes_path=None):
        self.model_path = model_path
        # Mini_XCEPTION luôn dùng 7 nhãn chuẩn này theo thứ tự
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.model = None
        
        # Hàng đợi để lưu 5 kết quả gần nhất -> Làm mượt (Smooth)
        self.emotion_window = deque(maxlen=5)
        
        # Cấu hình MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.load_resources()

    def load_resources(self):
        # Load Model
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                print(f"[INFO] Load model {self.model_path} thành công.")
            except Exception as e:
                print(f"[ERROR] Không tải được model: {e}")
        else:
            print(f"[WARNING] Không thấy file model tại {self.model_path}")

    def preprocess_input(self, face_image):
        """
        Chuẩn hóa ảnh theo chuẩn của Mini_XCEPTION:
        1. Resize 64x64
        2. Grayscale
        3. Normalize về khoảng [-1, 1] thay vì [0, 1]
        """
        # Resize
        face_image = cv2.resize(face_image, (64, 64))
        
        # Grayscale
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Normalize: Chuyển về float32 và đưa về khoảng [-1, 1]
        face_image = face_image.astype('float32') / 255.0
        face_image = (face_image - 0.5) * 2.0 
        
        # Expand dims: (64, 64) -> (1, 64, 64, 1)
        face_image = np.expand_dims(face_image, axis=-1)
        face_image = np.expand_dims(face_image, axis=0)
        
        return face_image

    def predict(self, frame):
        if self.model is None: return []

        results = []
        h_img, w_img, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = self.detector.process(img_rgb)

        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * w_img), int(bboxC.ymin * h_img), int(bboxC.width * w_img), int(bboxC.height * h_img)
                
                # --- CẢI TIẾN 1: THÊM PADDING (MỞ RỘNG VÙNG CẮT) ---
                # MediaPipe cắt rất sát, model cần nhìn rộng hơn một chút
                padding_x = int(w * 0.1) # Mở rộng 10% chiều ngang
                padding_y = int(h * 0.1) # Mở rộng 10% chiều dọc
                
                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w = min(w_img - x, w + 2 * padding_x)
                h = min(h_img - y, h + 2 * padding_y)
                # ---------------------------------------------------

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0: continue

                try:
                    # --- CẢI TIẾN 2: CHUẨN HÓA ĐÚNG CÁCH ---
                    roi = self.preprocess_input(face_crop)

                    # Dự đoán
                    preds = self.model.predict(roi, verbose=0)[0]

                    # --- CẢI TIẾN 3: ĐIỀU CHỈNH ĐỘ NHẠY (SENSITIVITY) ---
                    # Logic: Giảm điểm Neutral, Tăng điểm các cảm xúc khó nhận diện
                    
                    # Giảm Neutral xuống còn 50% sức mạnh
                    neutral_idx = self.classes.index('neutral')
                    preds[neutral_idx] *= 0.50 

                    # Tăng nhẹ điểm cho 'sad' và 'fear' (thường bị lẫn với neutral)
                    # Nếu bạn thấy nó nhạy quá thì giảm số 1.5 xuống 1.2
                    sad_idx = self.classes.index('sad')
                    fear_idx = self.classes.index('fear')
                    preds[sad_idx] *= 1.5 
                    preds[fear_idx] *= 1.2
                    
                    # ----------------------------------------------------

                    # Smoothing: Cộng dồn kết quả cũ để tránh nhảy số
                    self.emotion_window.append(preds)
                    avg_preds = np.mean(self.emotion_window, axis=0)
                    
                    max_idx = np.argmax(avg_preds)
                    emotion_label = self.classes[max_idx]
                    confidence = float(avg_preds[max_idx])

                    results.append({
                        'box': (x, y, w, h),
                        'emotion': emotion_label,
                        'score': confidence,
                        'all_scores': avg_preds
                    })
                except Exception as e:
                    print(f"Error prediction: {e}")

        return results