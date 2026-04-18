import tensorflow as tf
from cnn_blocks import modern_block, inception_block

class inception_model(tf.keras.Model) :
    def __init__(self, activation = "relu") :
        super().__init__()

        # Basic Feature Extractor 
        self.modern_block_1 = modern_block(filters = 32, kernel_size = 3, activation = activation, pool_size = 2)
        self.modern_block_2 = modern_block(filters = 64, kernel_size = 3, activation = activation, pool_size = 2)

        # Inception Blocks 1 
        self.inception_1 = inception_block(
            activation = activation,
            f1_reduce = 64,
            f3_reduce = 96, f3_expand = 128,
            f5_reduce = 16, f5_expand = 32
        )
        # Inception Block 2
        self.inception_2 = inception_block(
            activation = activation,
            f1_reduce = 64,
            f3_reduce = 96, f3_expand = 128,
            f5_reduce = 16, f5_expand = 32
        )
        # Downsample 
        self.pool_1      = tf.keras.layers.MaxPool2D(2)

        # Inception Block 3 
        self.inception_3 = inception_block(
            activation = activation,
            f1_reduce = 128,
            f3_reduce = 128, f3_expand = 192,
            f5_reduce = 32, f5_expand = 64
        )
        # Inception Block 4
        self.inception_4 = inception_block(
            activation = activation,
            f1_reduce = 128,
            f3_reduce = 128, f3_expand = 192,
            f5_reduce = 32, f5_expand = 64
        )
        # Downsample 
        self.pool_2      = tf.keras.layers.MaxPool2D(2)

        # Head 
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.drop        = tf.keras.layers.Dropout(0.3)
        self.classifier  = tf.keras.layers.Dense(units = 1, activation = "sigmoid")

    def call(self, x) :
        x = self.modern_block_2(self.modern_block_1(x))
        x = self.pool_1(self.inception_2(self.inception_1(x)))
        x = self.pool_2(self.inception_4(self.inception_3(x)))
        x = self.classifier(self.drop(self.global_pool(x)))
        return x 