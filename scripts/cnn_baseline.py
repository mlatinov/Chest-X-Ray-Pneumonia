import tensorflow as tf
from cnn_blocks import modern_block

class Baseline(tf.keras.Model) :
    def __init__(self, activation = "relu") :
        super().__init__()

        # Encoder 
        self.block1 = modern_block(filters = 16 ,kernel_size = 3 , activation = activation, pool_size = 2)
        self.block2 = modern_block(filters = 32 ,kernel_size = 3 , activation = activation, pool_size = 2)

        # Head 
        self.gap        = tf.keras.layers.GlobalAveragePooling2D()
        self.drop       = tf.keras.layers.Dropout(rate = 0.5)
        self.classifier = tf.keras.layers.Dense(units= 1, activation = "sigmoid")

    def call(self, x, training = False) :
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = self.drop(x, training = training) 
        x = self.classifier(x)
        return x 