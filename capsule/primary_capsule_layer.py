

import tensorflow as tf
from capsule.utils import squash

layers = tf.keras.layers
models = tf.keras.models


class PrimaryCapsule(tf.keras.Model):

    def __init__(self, channels=32, dim=8, kernel_size=(9, 9), strides=2, routing='conv', name=''):
        super(PrimaryCapsule, self).__init__(name=name)
        assert (channels % dim == 0) or (channels == 1), "Invalid size of channels and dim_capsule"

        self.channels = channels
        self.dim = dim
        self.routing = routing

        num_filters = channels * dim
        self.conv1 = layers.Conv2D(
            name="conv2d",
            filters=num_filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            kernel_initializer="he_normal",
            padding='valid')
        
        self.reshape = layers.Reshape(target_shape = (-1, dim))


    def call(self, inputs):
        x = self.conv1(inputs)

        if self.routing == 'conv':
            shape = x.shape
            shape = [shape[0], shape[1], shape[2], self.channels, self.dim]
            x = tf.reshape(x, shape=shape)
        else:
            x = self.reshape(x)

        return squash(x)
