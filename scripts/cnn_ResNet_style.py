import tensorflow as tf 
from cnn_blocks import modern_block , residual_block

class Resnet(tf.keras.Model) :
    def __init__(self, activation) :
        super().__init__()

        # Initial Feature Extraction 
        self.feature_extraction = modern_block(filters = 8, kernel_size = 3, activation = activation, pool_size = 2)

        # Learn Residuals 
        self.resid_block_1 = residual_block(filters = 16, kernel_size = 3, activation = activation)
        self.resid_block_2 = residual_block(filters = 16, kernel_size = 3, activation = activation)

        # Downsample 
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size = 2)

        # Learn Residuals 
        self.resid_block_3 = residual_block(filters = 32, kernel_size = 3, activation = activation)
        self.resid_block_4 = residual_block(filters = 32, kernel_size = 3, activation = activation)

        # Downsample 
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size = 2)

        # Head 
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.drop        = tf.keras.layers.Dropout(rate = 0.5)
        self.classifier  = tf.keras.layers.Dense(units = 1, activation = "sigmoid")

    def call(self, x , training = False) :
        x = self.feature_extraction(x)
        x = self.pool_1(self.resid_block_2(self.resid_block_1(x)))
        x = self.pool_2(self.resid_block_4(self.resid_block_3(x)))
        x = self.classifier(self.drop(self.global_pool(x)))
        return x 
