#Importing dependencies
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


#Resizes and normailzes the images
def preprocess(image, label, img_size):
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label



#Loading and creating a dataset
def load_cifar10(
    img_size=224,
    batch_size=32,
    buffer_size=10000
):
    #Loading CIFAR-10 from Keras
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    #Creating tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    #Training pipeline
    train_ds = (
        train_ds
        #Randomizes sample order in each epoch
        .shuffle(buffer_size)   
        #preprocesses images
        .map(
            lambda x, y: preprocess(x, y, img_size),     
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        #overlaps cpu preprocessing and gpu training
        .prefetch(tf.data.AUTOTUNE)
    )

    # Test pipeline
    test_ds = (
        test_ds
        .map(
            lambda x, y: preprocess(x, y, img_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds
