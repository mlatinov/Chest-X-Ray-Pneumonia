import mlflow
import tensorflow as tf 
from scripts.cnn_Inception_style import inception_model
from scripts.process_image import process_data
from scripts.plot_history import plot_history

def exp_inception() :
    # Automaticly Log model metrics 
    mlflow.keras.autolog()

    with mlflow.start_run(run_name = "Inception"):
        # Process data 
        model_data = process_data(
            dir_train_path      = "sample_data_img/train/",
            dir_validation_path = "sample_data_img/test/",
            img_size            = (244, 244)
        )
        # Get the model 
        inception = inception_model(activation = "relu")

        # Compile the model 
        inception.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss      = tf.keras.losses.BinaryCrossentropy(),
            metrics   = [
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision()
            ] 
        )
        # Fit the model 
        history = inception.fit(
            x               = model_data["train_samples"],
            validation_data = model_data["val_samples"],
            epochs          = 20,
            callbacks       = [
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
