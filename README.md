# AI Face System (Verification & Emotion)

## Cài đặt
1. Yêu cầu: Python 3.9+, CUDA 12.4 (nếu dùng GPU).
2. Cài thư viện: `pip install -r requirements.txt`

## Dữ liệu
1. Tải FER-2013 và đặt vào `dataset/fer2013`.
2. Đặt ảnh người dùng cần nhận diện vào `database/`. Tên file là tên người dùng.

## Huấn luyện (Emotion)
Chạy: `python train_emotion.py`

## Testing
1. Test Điểm danh: `python tests/test_face_verification.py` (Nhấn 'v' để xác thực).
2. Test Cảm xúc: `python tests/test_face_emotion.py`.