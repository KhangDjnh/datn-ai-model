import sys
import os
import cv2

# Thêm thư mục gốc vào path để import được module 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.face_verification import FaceVerifier

def main():
    # Khởi tạo core logic
    verifier = FaceVerifier(db_path="database")
    
    cap = cv2.VideoCapture(0)
    print("=== FACE VERIFICATION TEST ===")
    print("Nhấn 'v' để xác thực | Nhấn 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()

        # Vẽ hướng dẫn
        cv2.putText(display_frame, "Press 'v' to Verify", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        
        # Chỉ khi nhấn 'v' mới gọi hàm xử lý (tiết kiệm tài nguyên)
        if key & 0xFF == ord('v'):
            print("Đang xử lý...")
            name, distance = verifier.verify(frame)
            
            if name:
                print(f"-> Chào mừng: {name} (Độ sai số: {distance:.4f})")
                cv2.putText(display_frame, f"HELLO: {name}", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                # Dừng hình 1 chút để nhìn kết quả
                cv2.imshow("Test Verification", display_frame)
                cv2.waitKey(2000) 
            else:
                print("-> Không nhận diện được người dùng.")
                cv2.putText(display_frame, "UNKNOWN", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("Test Verification", display_frame)
                cv2.waitKey(1000)

        cv2.imshow("Test Verification", display_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()