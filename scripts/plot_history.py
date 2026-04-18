
import matplotlib.pyplot as plt

def plot_history(history) : 
    
    fig, axes = plt.subplots(1, 3, figsize = (15, 4))

    # Loss plot 
    axes[0].plot(history.history["loss"],     label = "Training Loss")
    axes[0].plot(history.history["val_loss"], label = "Validation Loss")
    axes[0].set_title("Loss")
    axes[0].legend() 

    # Auc 
    axes[1].plot(history.history["auc_1"],     label = "Traning AUC")
    axes[1].plot(history.history["val_auc_1"], label = "Validation AUC")
    axes[1].set_title("AUC")
    axes[1].legend()

    # Precision / Recall 
    axes[2].plot(history.history["precision_1"],     label = "Traning Precision")
    axes[2].plot(history.history["val_precision_1"], label = "Validaiton Precision")
    axes[2].plot(history.history["recall_1"],        label = "Training Recall")
    axes[2].plot(history.history["val_recall_1"],    label = "Validation Racall")
    axes[2].set_title("Precision / Recall")
    axes[2].legend()

    plt.tight_layout()

    return fig
