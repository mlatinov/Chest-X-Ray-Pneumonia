
import mlflow
import tensorflow as tf
from scripts.cnn_baseline import Baseline
from scripts.process_image import process_data
from scripts.plot_history import plot_history

def exp_baseline_cnn() :
    # Log Everything 
    mlflow.tensorflow.autolog()

    # Start the experiment 
    with mlflow.start_run(run_name = "Baseline CNN") :

        # Process data 
        model_data = process_data(
            dir_test_path       = "sample_data_img/test/",
            dir_train_path      = "sample_data_img/train/",
            dir_validation_path = "sample_data_img/val/",
            img_size            = (224, 224)
        )
        # Get the model 
        baseline_model = Baseline(activation = "relu")

        # Compile the model 
        baseline_model.compile(
            loss      = tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics   = [
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision()
            ]
        )
        # Fit the model 
        history =  baseline_model.fit(
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
                    patience = 2,
                    min_lr   = 1e-6,
                    verbose  = 1 
                )
            ]
        )
        # Ploting and logging the metrics 
        fig = plot_history(history)
        mlflow.log_figure(fig, "training_curves.png")

        # Prediction on the test set 
        model_eval = baseline_model.evaluate(x = model_data["test_data"],return_dict = True) 

        # Log the model evaluation on the test set 
        mlflow.log_metrics(model_eval)
