# core/emotion_recognition.py
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

class EmotionRecognizer:
    def __init__(self, model_path='models/emotion_model.h5', classes_path='models/classes.txt'):
        self.model_path = model_path
        self.classes_path = classes_path
        self.classes = []
        self.model = None
        
        # Cấu hình MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.load_resources()

    def load_resources(self):
        """Load model và danh sách nhãn cảm xúc"""
        if os.path.exists(self.classes_path):
            with open(self.classes_path, 'r') as f:
                self.classes = f.read().splitlines()
        else:
            print(f"[WARNING] Không tìm thấy file {self.classes_path}")

        if os.path.exists(self.model_path):
            print("[INFO] Đang load model cảm xúc...")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print("[INFO] Load model thành công.")
            except Exception as e:
                print(f"[ERROR] Không thể load model: {e}")
        else:
            print(f"[WARNING] Không tìm thấy model tại {self.model_path}. Hãy train model trước.")

    def predict(self, frame):
        """
        Nhận diện cảm xúc từ frame hình ảnh.
        :param frame: Ảnh BGR từ OpenCV.
        :return: List các kết quả [{'box': [x,y,w,h], 'emotion': str, 'score': float}, ...]
        """
        if self.model is None:
            return []

        results = []
        h_img, w_img, _ = frame.shape
        
        # MediaPipe cần ảnh RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = self.detector.process(img_rgb)

        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w_img)
                y = int(bboxC.ymin * h_img)
                w = int(bboxC.width * w_img)
                h = int(bboxC.height * h_img)

                # Fix toạ độ nếu nó bị âm hoặc trồi ra ngoài ảnh
                x, y = max(0, x), max(0, y)
                w, h = min(w_img - x, w), min(h_img - y, h)

                # Cắt mặt
                face_crop = frame[y:y+h, x:x+w]
                
                if face_crop.size == 0:
                    continue

                try:
                    # --- SỬA LỖI TẠI ĐÂY: Đổi 96 thành 48 ---
                    roi = cv2.resize(face_crop, (48, 48)) 
                    # ----------------------------------------
                    
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) # Custom CNN của mình train bằng RGB
                    roi = np.expand_dims(roi, axis=0) # Thêm dimension batch: (1, 48, 48, 3)
                    roi = roi / 255.0 # Chuẩn hóa về 0-1 giống lúc train

                    # Dự đoán
                    preds = self.model.predict(roi, verbose=0)[0]
                    max_idx = np.argmax(preds)
                    
                    emotion_label = self.classes[max_idx] if self.classes else str(max_idx)
                    confidence = float(preds[max_idx])

                    results.append({
                        'box': (x, y, w, h),
                        'emotion': emotion_label,
                        'score': confidence
                    })
                except Exception as e:
                    print(f"[ERROR] Processing face: {e}")

        return results