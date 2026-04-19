import tensorflow as tf
from cnn_blocks import unet_block, inception_block

class Unet_inception(tf.keras.Model) :
    def __init__(self, activation) :
        super().__init__()

        #### Model Architechture ####

        # Unet Block 
        self.unet_1 = unet_block(filters=16, dropout_rate=0.4, activation=activation)
        self.pool_1 = tf.keras.layers.MaxPool2D(2)

        self.unet_2 = unet_block(filters=32, dropout_rate=0.4, activation=activation)
        self.pool_2 = tf.keras.layers.MaxPool2D(2)

        self.unet_3 = unet_block(filters=64, dropout_rate=0.4, activation=activation)
        self.pool_3 = tf.keras.layers.MaxPool2D(2)

        # Inception Block 
        self.inception_1 = inception_block(
            f1_reduce  =16,
            f3_reduce  =16, f3_expand=32,
            f5_reduce  =8,  f5_expand=16,
            activation = activation
        )

        # Classification Head 
        self.gap        = tf.keras.layers.GlobalAveragePooling2D()
        self.drop       = tf.keras.layers.Dropout(0.4)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x) :
        
        # Encoder
        x = self.pool_1(self.unet_1(x))  
        x = self.pool_2(self.unet_2(x))   
        x = self.pool_3(self.unet_3(x))   

        # Inception Bottleneck
        x = self.inception_1(x)            

        # Head
        x = self.gap(x)                    
        x = self.drop(x)
        return self.classifier(x)
