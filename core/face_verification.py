# core/face_verification.py
import os
import cv2
import pandas as pd
from deepface import DeepFace

class FaceVerifier:
    def __init__(self, db_path="database", model_name="ArcFace", detector_backend="mediapipe"):
        """
        Khởi tạo module xác thực khuôn mặt.
        :param db_path: Đường dẫn tới thư mục chứa ảnh gốc.
        :param model_name: Tên model nhận diện (khuyên dùng ArcFace hoặc VGG-Face).
        :param detector_backend: Backend để cắt mặt (MediaPipe là nhanh nhất).
        """
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        
        # Kiểm tra thư mục database
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            print(f"[WARNING] Thư mục '{self.db_path}' chưa có ảnh nào. Hãy thêm ảnh vào đó.")

    def verify(self, img_frame):
        """
        Xác thực khuôn mặt trong frame hình ảnh.
        :param img_frame: Ảnh dạng numpy array (OpenCV Image).
        :return: (name, distance) hoặc (None, None) nếu không tìm thấy.
        """
        try:
            # DeepFace.find trả về danh sách các DataFrame
            dfs = DeepFace.find(
                img_path=img_frame,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False, # Không báo lỗi nếu không thấy mặt
                silent=True,
                align=True
            )

            if len(dfs) > 0:
                # Lấy kết quả đầu tiên
                df = dfs[0]
                if not df.empty:
                    # Sắp xếp theo độ chính xác (distance càng nhỏ càng giống)
                    df = df.sort_values(by=['distance'])
                    best_match = df.iloc[0]
                    
                    identity_path = best_match['identity']
                    distance = best_match['distance']
                    
                    # Lấy tên file làm tên người dùng (bỏ đường dẫn và đuôi file)
                    # Ví dụ: database/NguyenVanA.jpg -> NguyenVanA
                    name = os.path.basename(identity_path).split('.')[0]
                    
                    return name, distance
            
            return None, None

        except Exception as e:
            print(f"[ERROR] Face Verification: {e}")
            return None, None