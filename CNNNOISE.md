import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# creating a sequential model
cnn = tf.keras.models.Sequential()

# adding convolution layer to network
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# adding pooling layer to network
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
