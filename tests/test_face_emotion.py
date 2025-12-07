import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.emotion_recognition import EmotionRecognizer

def main():
    emotion_recognizer = EmotionRecognizer()
    classes = emotion_recognizer.classes # Lấy danh sách class

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        results = emotion_recognizer.predict(frame)

        for res in results:
            x, y, w, h = res['box']
            emotion = res['emotion']
            score = res['score']
            all_scores = res.get('all_scores', [])

            # Vẽ khung mặt
            color = (0, 255, 0)
            if emotion in ['angry', 'fear', 'sad', 'disgust']: color = (0, 0, 255) # Màu đỏ nếu tiêu cực
            if emotion == 'happy': color = (0, 255, 255) # Màu vàng nếu vui
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Viết tên cảm xúc chính
            text = f"{emotion} ({score:.0%})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # --- VẼ BIỂU ĐỒ THANH (BAR CHART) BÊN CẠNH ---
            # Để xem model đang nghĩ gì về các cảm xúc khác
            if len(all_scores) > 0 and len(classes) > 0:
                bar_x = x + w + 10 # Vẽ bên phải khuôn mặt
                bar_y = y
                
                for i, prob in enumerate(all_scores):
                    label = classes[i]
                    # Vẽ thanh nền
                    cv2.rectangle(frame, (bar_x, bar_y + i*20), (bar_x + 100, bar_y + i*20 + 15), (200, 200, 200), -1)
                    # Vẽ thanh giá trị
                    bar_length = int(prob * 100)
                    cv2.rectangle(frame, (bar_x, bar_y + i*20), (bar_x + bar_length, bar_y + i*20 + 15), (0, 165, 255), -1)
                    # Tên label
                    cv2.putText(frame, f"{label[:3]}", (bar_x - 35, bar_y + i*20 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Test Emotion (Smart)", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()