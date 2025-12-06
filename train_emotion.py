# train_emotion.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# --- CẤU HÌNH ---
IMG_SIZE = 48
BATCH_SIZE = 64 # Custom CNN nhẹ nên tăng batch size lên 64 cho train nhanh hơn
EPOCHS = 50
DATA_DIR = 'dataset/fer2013'
MODEL_PATH = 'models/emotion_model.h5'

def build_custom_cnn(num_classes):
    """
    Mô hình CNN lấy cảm hứng từ VGG/Mini-Xception, tối ưu cho ảnh 48x48
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten & Dense
    model.add(Flatten())
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes, activation='softmax'))

    return model

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    # Data Augmentation: Tăng cường đa dạng dữ liệu
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.15,       # Thêm zoom
        width_shift_range=0.1, # Thêm dịch chuyển
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("Đang load dữ liệu...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"Classes: {class_names}")

    # Xây dựng model
    model = build_custom_cnn(num_classes)
    
    # Compile
    # Dùng learning rate nhỏ 0.0005 để học kỹ hơn
    opt = Adam(learning_rate=0.0005) 
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary() # In cấu trúc model ra xem

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=8, # Kiên nhẫn hơn chút
        restore_best_weights=True,
        verbose=1
    )
    
    # Tự giảm learning rate khi loss không giảm
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=4, 
        min_lr=0.00001,
        verbose=1
    )

    print("Bắt đầu train Custom CNN...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    model.save(MODEL_PATH)
    print(f"Đã lưu model tại {MODEL_PATH}")

    with open('models/classes.txt', 'w') as f:
        f.write('\n'.join(class_names))

if __name__ == "__main__":
    main()