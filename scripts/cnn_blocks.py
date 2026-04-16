import tensorflow as tf 

class clasical_block(tf.keras.layers.Layer) :
    """
    Define Classical Block as : Conv2D → Activation → MaxPooling2D
    """
    def __init__(self, filters, kernel_size, activation, pool_size) :
        super().__init__()

        # Define the Network Block 
        self.conv2d     = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = "same")
        self.activation = tf.keras.layers.Activation(activation = activation)
        self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size = pool_size)
    
    def call(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        x = self.max_pool2d(x)
        return x

class modern_block(tf.keras.layers.Layer) :
    """
    Modern CNN Block Conv2D → BatchNormalization → Activation -> MaxPooling2D
    """
    def __ini__(self, filters, kernel_size, activation, pool_size) :
        super().__init__()

        # Define the Network Block
        self.conv2d     = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = "same")
        self.bn         = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation = activation)
        self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size = pool_size)

    def call(self, x) :
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.max_pool2d(x)
        return x 

class residual_block(tf.keras.layers.Layer) :
    """
    Modern Residual Block Conv → BN → Activation → Conv → BN → Add(input) → Activation
    """
    def __init__(self, filters, kernel_size, activation) :
        super().__init__()

        # Define the Network Parts 
        self.conv2d_1     = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = "same")
        self.bn_1         = tf.keras.layers.BatchNormalization()
        self.activation   = tf.keras.layers.Activation(activation = activation)
        self.conv2d_2     = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = "same")
        self.bn_2         = tf.keras.layers.BatchNormalization() 

    def call(self, x, training = False) :
        # Branch in the begginng
        branch = x 

        # Block 
        x = self.conv2d_1(x)
        x = self.bn_1(x, training = training)
        x = self.activation(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x, training = training)

        # Merge the brach 
        x = x + branch
        x = tf.keras.layers.Activation(x)
        return x 

class inception_block(tf.keras.layers.Layer) :
    """
    Inception block containing 3 Branches with 1x1 3x3 5x5 
    General Structure : of each Branch : Conv2D → BatchNormalization → Activation 

    """
    def __init__(
        self,
        f1_reduce,
        f3_reduce,f3_expand,
        f5_reduce,f5_expand,
        activation
        ) :
        super().__init__()
        # Define Inception Block with 3 branches

        # 1x1 Block 
        self.inception_1x1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = f1_reduce, kernel_size = 1, padding = "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation = activation)
        ])

        # 1x1 -> 3x3 Block 
        self.inception_3x3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = f3_reduce, kernel_size = 1, padding = "same"), 
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation = activation),
            tf.keras.layers.Conv2D(filters = f3_expand, kernel_size= 3, padding = "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation = activation)
        ])

        # 1x1 -> 5x5 Block 
        self.inception_5x5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = f5_reduce, kernel_size = 1, padding = "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation = activation),
            tf.keras.layers.Conv2D(filters = f5_expand, kernel_size = 5, padding = "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation = activation)
        ])
    def call(self, x) :
        # Brach 
        b1 = self.inception_1x1(x)
        b2 = self.inception_3x3(x)
        b3 = self.inception_5x5(x)

        # Merge the Branches 
        x = tf.concat([b1, b2, b3], axis = -1 )
        return x 

class unet_block(tf.keras.layers.Layer) : 
    """
    Unet Block 2x(Conv2D -> BatchNormalization -> Activation) ->  Dropout
    """
    def __init__(self, filters, dropout_rate, activation) : 
        super().__init__()

        # Define the Unet Block 
        self.conv2d_1     = tf.keras.layers.Conv2D(filters = filters, kernel_size = 3, padding = "same")
        self.bn_1         = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.Activation(activation = activation)
        self.conv2d_2     = tf.keras.layers.Conv2D(filters = filters, kernel_size = 3, padding = "same")
        self.bn_2         = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.Activation(activation = activation) 
        self.drop         = tf.keras.layers.Dropout(rate = dropout_rate)

    def call(self, x) :
        x = self.activation_1(self.bn_1(self.conv2d_1(x))) # Conv -> BatchNormalization -> Activation 
        x = self.activation_2(self.bn_2(self.conv2d_2(x))) # Conv -> BatchNormalization -> Activation
        x = self.drop(x)
        return x 