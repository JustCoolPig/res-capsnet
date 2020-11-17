

import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

class Residual(tf.keras.Model):
    def call(self, out_prev, out_skip):
        x = tf.keras.layers.Add()([out_prev, out_skip])
        # TODO call squash-function here
        return x
