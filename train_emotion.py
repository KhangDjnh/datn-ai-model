# train_emotion.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Cấu hình
IMG_SIZE = 48  # FER2013 là 48x48, nhưng MobileNet thích ảnh to hơn chút, ta sẽ resize
INPUT_SHAPE = (96, 96, 3) # Resize lên 96 để MobileNet hoạt động tốt hơn
BATCH_SIZE = 32 # Với GTX 1650 4GB, để 32 hoặc 16
EPOCHS = 20
DATA_DIR = 'dataset/fer2013'
MODEL_PATH = 'models/emotion_model.h5'

def build_model(num_classes):
    # Dùng MobileNetV2 bỏ phần đầu (include_top=False)
    base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
    
    # Đóng băng các lớp cơ sở để không train lại từ đầu (nhanh hơn)
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x) # Tránh overfitting
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    # Data Augmentation (Làm phong phú dữ liệu)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load dữ liệu (Lưu ý: chuyển về RGB vì MobileNet cần 3 kênh màu)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(96, 96),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb' 
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(96, 96),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb'
    )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"Classes: {class_names}")

    # Xây dựng model
    model = build_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    print("Bắt đầu train...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Lưu model
    model.save(MODEL_PATH)
    print(f"Đã lưu model tại {MODEL_PATH}")

    # Lưu tên class để dùng lại
    with open('models/classes.txt', 'w') as f:
        f.write('\n'.join(class_names))

if __name__ == "__main__":
    main()