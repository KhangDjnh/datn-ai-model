import sys
import os
import cv2

# Thêm thư mục gốc vào path để import được module 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.emotion_recognition import EmotionRecognizer

def main():
    # Khởi tạo core logic
    emotion_recognizer = EmotionRecognizer()
    
    cap = cv2.VideoCapture(0)
    print("=== EMOTION RECOGNITION TEST ===")
    print("Nhấn 'Esc' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Gọi hàm xử lý từ core
        results = emotion_recognizer.predict(frame)

        # Vẽ kết quả lên màn hình
        for res in results:
            x, y, w, h = res['box']
            emotion = res['emotion']
            score = res['score']

            # Vẽ khung chữ nhật
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Viết chữ cảm xúc
            text = f"{emotion} ({score:.0%})"
            cv2.putText(frame, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Test Emotion", frame)

        if cv2.waitKey(1) & 0xFF == 27: # Esc
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()