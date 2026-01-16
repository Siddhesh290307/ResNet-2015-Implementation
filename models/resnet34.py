#Importing dependencies
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D,BatchNormalization,Activation,MaxPooling2D,GlobalAveragePooling2D,Dense)

#Importing residualblock
from .residual_block import ResidualBlock





class ResNet34(Model):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__(name='resnet34')

     
        self.stem_conv = Conv2D(64, 7, strides=2, padding='same', use_bias=False)
        self.stem_bn = BatchNormalization()
        self.stem_relu = Activation('relu')
        self.stem_pool = MaxPooling2D(pool_size=3, strides=2, padding='same')


        self.conv2_1 = ResidualBlock(64)
        self.conv2_2 = ResidualBlock(64)
        self.conv2_3 = ResidualBlock(64)

        self.conv3_1 = ResidualBlock(128, stride=2)
        self.conv3_2 = ResidualBlock(128)
        self.conv3_3 = ResidualBlock(128)
        self.conv3_4 = ResidualBlock(128)

    
        self.conv4_1 = ResidualBlock(256, stride=2)
        self.conv4_2 = ResidualBlock(256)
        self.conv4_3 = ResidualBlock(256)
        self.conv4_4 = ResidualBlock(256)
        self.conv4_5 = ResidualBlock(256)
        self.conv4_6 = ResidualBlock(256)

        self.conv5_1 = ResidualBlock(512, stride=2)
        self.conv5_2 = ResidualBlock(512)
        self.conv5_3 = ResidualBlock(512)

     
        self.avg_pool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
   
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_relu(x)
        x = self.stem_pool(x)


        x = self.conv2_1(x, training=training)
        x = self.conv2_2(x, training=training)
        x = self.conv2_3(x, training=training)


        x = self.conv3_1(x, training=training)
        x = self.conv3_2(x, training=training)
        x = self.conv3_3(x, training=training)
        x = self.conv3_4(x, training=training)

        x = self.conv4_1(x, training=training)
        x = self.conv4_2(x, training=training)
        x = self.conv4_3(x, training=training)
        x = self.conv4_4(x, training=training)
        x = self.conv4_5(x, training=training)
        x = self.conv4_6(x, training=training)


        x = self.conv5_1(x, training=training)
        x = self.conv5_2(x, training=training)
        x = self.conv5_3(x, training=training)

        x = self.avg_pool(x)
        return self.fc(x)
