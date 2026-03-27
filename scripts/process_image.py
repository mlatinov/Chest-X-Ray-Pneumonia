def process_data(dir_train_path, dir_validation_path, img_size):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    validation_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        directory=dir_train_path,
        batch_size=32,
        target_size=img_size,
        class_mode="binary",
        seed=42
    )
    validation_data = validation_gen.flow_from_directory(
        directory=dir_validation_path,
        batch_size=32,
        target_size=img_size,
        class_mode="binary",
        seed=42
    )

    # 👇 ONLY return primitive / serializable stuff
    return {
        "train_dir": dir_train_path,
        "val_dir": dir_validation_path,
        "img_size": img_size,
        "train_samples": train_data.samples,
        "val_samples": validation_data.samples
    }