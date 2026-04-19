
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def xai_grad_cam(model, test_data, n_samples=4):

    test_data.reset()
    images, labels = next(iter(test_data))
    images = images[:n_samples]
    labels = labels[:n_samples]

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 4 * n_samples))

    for i in range(n_samples):
        img = images[i:i+1]  

        # Find the last conv layer output
        last_conv_model = tf.keras.Model(
            inputs  = model.input,
            outputs = [
                [l.output for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1],
                model.output
            ]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_output, pred = last_conv_model(img, training=False)
            loss = pred[:, 0]

        grads   = tape.gradient(loss, conv_output)         # gradients w.r.t last conv
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))    # average over spatial dims
        cam     = tf.reduce_sum(conv_output[0] * weights, axis=-1)  # weighted sum
        cam     = tf.maximum(cam, 0)                       # ReLU
        cam     = cam / (tf.reduce_max(cam) + 1e-8)        # normalize 

        # Resize heatmap to image size
        heatmap = tf.image.resize(cam[..., tf.newaxis], (images.shape[1], images.shape[2]))
        heatmap = np.squeeze(heatmap.numpy())

        # Plot
        raw_img    = np.squeeze(images[i])
        pred_score = float(pred[0][0])
        pred_label = "Pneumonia" if pred_score > 0.5 else "Normal"
        true_label = "Pneumonia" if int(labels[i]) == 1 else "Normal"
        color      = "green" if pred_label == true_label else "red"

        axes[i][0].imshow(raw_img, cmap="gray")
        axes[i][0].set_title("Original CT")
        axes[i][0].axis("off")

        axes[i][1].imshow(heatmap, cmap="jet")
        axes[i][1].set_title("GradCAM Heatmap")
        axes[i][1].axis("off")

        axes[i][2].imshow(raw_img, cmap="gray")
        axes[i][2].imshow(heatmap, cmap="jet", alpha=0.4)
        axes[i][2].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_score:.2f})", color=color)
        axes[i][2].axis("off")

    plt.suptitle("GradCAM — Model Attention on CT Scans", fontsize=14)
    plt.tight_layout()
    return fig
