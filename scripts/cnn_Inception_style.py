import tensorflow as tf
from cnn_blocks import modern_block, inception_block

class inception_model(tf.keras.Model):
    def __init__(self, activation="relu"):
        super().__init__()

        # Stem — 16ch out
        self.modern_block_1 = modern_block(filters=8,  kernel_size=3, activation=activation, pool_size=2)
        self.modern_block_2 = modern_block(filters=16, kernel_size=3, activation=activation, pool_size=2)

        # Stage 1 — both blocks output 32ch (8+16+8=32)
        self.inception_1 = inception_block(
            activation=activation,
            f1_reduce=8,
            f3_reduce=8,  f3_expand=16,
            f5_reduce=4,  f5_expand=8
        )
        self.inception_2 = inception_block(
            activation=activation,
            f1_reduce=8,
            f3_reduce=8,  f3_expand=16,
            f5_reduce=4,  f5_expand=8
        )
        self.pool_1 = tf.keras.layers.MaxPool2D(2)

        # Stage 2 — both blocks output 64ch (16+32+16=64)
        self.inception_3 = inception_block(
            activation=activation,
            f1_reduce=16,
            f3_reduce=16, f3_expand=32,
            f5_reduce=8,  f5_expand=16
        )
        self.inception_4 = inception_block(
            activation=activation,
            f1_reduce=16,
            f3_reduce=16, f3_expand=32,
            f5_reduce=8,  f5_expand=16
        )
        self.pool_2 = tf.keras.layers.MaxPool2D(2)

        # Head
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.drop        = tf.keras.layers.Dropout(0.4)
        self.classifier  = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.modern_block_2(self.modern_block_1(x))
        x = self.pool_1(self.inception_2(self.inception_1(x)))
        x = self.pool_2(self.inception_4(self.inception_3(x)))
        x = self.classifier(self.drop(self.global_pool(x)))
        return x