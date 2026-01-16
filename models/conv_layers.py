#Importing dependencies
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation 


#Creating a custom convolution layer
class CustomConv2D(Layer):
    def __init__(self, n_filters, kernel_size, n_strides, padding='valid', activate=True):
        super().__init__()
        
        #activate is a variable used to check if relu is to be applied or not
        self.activate = activate 

        self.conv= Conv2D(n_filters, kernel_size, strides=n_strides, padding=padding, use_bias=False)
        self.bn=BatchNormalization()
        self.activation=Activation('relu')

    def call(self, x, training=False):
        x=self.conv(x)
        x=self.bn(x,training=training)

        #Checking if we should apply relu activation or not
        if self.activate: 
            x=self.activation(x)

        return x
