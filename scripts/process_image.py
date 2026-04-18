
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def process_data(dir_train_path, dir_validation_path, img_size):

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,            
        zoom_range=0.1,               
        horizontal_flip=True,         
        width_shift_range=0.05,       
        height_shift_range=0.05,
        brightness_range=[0.8, 1.2],  #  scanner variability
    )
    validation_gen = ImageDataGenerator(rescale=1./255)  # no augmentation on val

    train_data = train_gen.flow_from_directory(
        directory=dir_train_path,
        batch_size=32,
        target_size=img_size,
        color_mode='grayscale',  # CT scans are single channel
        class_mode='binary',
        seed=42
    )
    validation_data = validation_gen.flow_from_directory(
        directory=dir_validation_path,
        batch_size=32,
        target_size=img_size,
        color_mode='grayscale',       
        class_mode='binary',
        seed=42
    )
    return {
        "train_samples": train_data,
        "val_samples": validation_data
    }