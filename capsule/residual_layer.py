

import tensorflow as tf
from capsule.utils import squash

layers = tf.keras.layers
models = tf.keras.models

class Residual(tf.keras.Model):
    def call(self, out_prev, out_skip):
        x = tf.keras.layers.Add()([out_prev, out_skip])
        #x = squash(x)
        return x
