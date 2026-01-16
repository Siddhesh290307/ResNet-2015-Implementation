#Importing dependencies
from tensorflow.keras.layers import Layer, Activation, Add

# Importing custom convolution block
from .conv_layers import CustomConv2D

#A single complete Residual Block
class ResidualBlock(Layer):
    def __init__(self,n_channels,  stride=1):
        super(ResidualBlock, self).__init__()


        #1st Convolution Layer
        self.conv1 = CustomConv2D(n_channels, kernel_size=3, n_strides=stride, padding='same')  

        #2nd Convolution Layer( relu activation is not run in this)
        self.conv2 = CustomConv2D(n_channels, kernel_size=3, n_strides=1,padding='same', activate=False)

        #projection layer is needed only if stride is not equal to 1
        self.use_projection = (stride != 1)

        if self.use_projection:
            self.projection = CustomConv2D(n_channels, kernel_size=1, n_strides=stride, padding='same')

        self.add = Add()
        self.activation = Activation('relu')

    def call(self, x, training=False):
        shortcut= x

        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)

        if self.use_projection:
            shortcut = self.projection(shortcut, training=training)

        #H(x)=F(x)+x
        x= self.add([x, shortcut])
        x= self.activation(x)

        return x
    


