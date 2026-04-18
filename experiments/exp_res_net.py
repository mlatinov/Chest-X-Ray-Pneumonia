import mlflow
import tensorflow as tf 
from scripts.cnn_ResNet_style import Resnet
from scripts.process_image import process_data 
from scripts.plot_history import plot_history
def exp_resnet() :

    # Autolog Keras Model Paramters 
    mlflow.keras.autolog()

    with mlflow.start_run(run_name = "ResNet"):
        # Process the data 
        model_data = process_data(
            dir_train_path      = "sample_data_img/train/",
            dir_validation_path = "sample_data_img/val/",
            img_size            = (244, 244) 
        )
        # Get the moedl 
        resnet = Resnet(activation = "relu")

        # Compile the model 
        resnet.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss      = tf.keras.losses.BinaryCrossentropy(),
            metrics   = [
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        # Train the model 
        history = resnet.fit(
            x                = model_data["train_samples"],
            validation_data  = model_data["val_samples"],
            epochs = 20,
            callbacks        = [
                tf.keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = 3,
                    restore_best_weights = True,
                    verbose              = 1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor  = "val_loss",
                    factor   = 0.5,
                    patience = 3,
                    min_lr   = 1e-6,
                    verbose  = 1
                )
            ]
        )
        # Ploting and logging the metrics 
        fig = plot_history(history)
        mlflow.log_figure(fig, "training_curves.png")
