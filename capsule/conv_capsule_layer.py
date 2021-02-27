

import tensorflow as tf
import capsule.utils as utils
from tensorflow.python.keras.utils import conv_utils
from capsule.utils import squash
import numpy as np

layers = tf.keras.layers
initializers = tf.keras.initializers

class ConvCapsule(layers.Layer):
    '''
    This class is largely based on the implementation of convolutional capsules in:
        - https://github.com/brjathu/deepcaps, and
        - https://github.com/amobiny/Deep_CapsNet
    '''

    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, routing_iterations=2, routing='dynamic',
            kernel_size=3, stride=1, padding='same', name=''):
        super(ConvCapsule, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.num_caps = in_capsules
        self.out_caps = out_capsules
        self.caps_dim = in_dim
        self.out_dim = out_dim
        self.strides = stride
        self.padding = padding
        self.iterations = routing_iterations
        self.routing = routing
        self.w_init = tf.random_normal_initializer(stddev=0.2)
        self.b_init = tf.constant_initializer(0.1)


    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"

        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.num_in_caps = input_shape[3]
        self.in_caps_dim = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                        self.in_caps_dim, self.num_caps * self.caps_dim],
                                 initializer=self.w_init,
                                 trainable=True,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_caps, self.caps_dim],
                                 initializer=self.b_init,
                                 trainable=True,
                                 name='b')
        self.built = True


    def call(self, input_tensor, training=None):
        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = tf.shape(input_transposed)
        input_tensor_reshaped = tf.reshape(input_transposed, [
            input_shape[0] * input_shape[1], self.input_height, self.input_width, self.in_caps_dim])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.in_caps_dim))

        conv = tf.nn.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                        padding=self.padding.upper(), data_format='NHWC')

        votes_shape = conv.shape
        _, conv_height, conv_width, _ = conv.get_shape()

        votes = tf.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                 self.num_caps, self.caps_dim])
        votes.set_shape((None, self.num_in_caps, conv_height, conv_width,
                         self.num_caps, self.caps_dim))

        logit_shape = tf.stack([input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_caps])
        biases_replicated = tf.tile(self.b, [conv_height, conv_width, 1, 1])

        if self.routing == 'rba':
            activations = dynamic_routing(votes, biases_replicated, logit_shape, self.num_in_caps, self.iterations)
        elif self.routing == 'sda':
            activations = sda_routing(votes, biases_replicated, logit_shape, self.num_in_caps, self.out_caps, self.iterations)
        else:
            raise NotImplementedError('Not implemented')
        return activations


# scaled distance agreement routing
def sda_routing(votes, biases, logit_shape, input_dim, out_capsules, num_routing):

    votes_t_shape = [5, 0, 1, 2, 3, 4]
    r_t_shape = [1, 2, 3, 4, 5, 0]
    votes_trans = tf.transpose(votes, votes_t_shape)

    # TODO  Ensure that ||u_hat|| <= ||v_i||

    def _body(i, logits, activations):
        """Routing while loop."""

        route = tf.nn.softmax(logits, axis=-1)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = squash(preactivate)
        activations = activations.write(i, activation)

        act_3d = tf.expand_dims(activation, 1)
        act_replicated = tf.tile(act_3d, [1, input_dim, 1, 1, 1, 1])
        
        # Calculate scale factor t
        p_p = 0.9
        d = tf.norm(act_replicated - votes, axis=-1, keepdims=True)
        d_o = tf.reduce_mean(tf.reduce_mean(d))
        d_p = d_o * 0.5
        t = tf.constant(np.log(p_p * (out_capsules - 1)) - np.log(1 - p_p), dtype=tf.float32) \
                / (d_p - d_o + 1e-12)
        t = tf.expand_dims(t, axis=-1)

        # Calc log prior using inverse distances
        logits = tf.norm(t * d, axis=-1)

        return i + 1, logits, activations

    activations = tf.TensorArray(
      dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
      lambda i, logits, activations: i < num_routing,
      _body,
      loop_vars=[i, logits, activations],
      swap_memory=True)

    return tf.cast(activations.read(num_routing - 1), dtype='float32')


# routing by agreement
def dynamic_routing(votes, biases, logit_shape, input_dim, num_routing):

    votes_t_shape = [5, 0, 1, 2, 3, 4]
    r_t_shape = [1, 2, 3, 4, 5, 0]
    votes_trans = tf.transpose(votes, votes_t_shape)

    def _body(i, logits, activations):
        """Routing while loop."""

        route = tf.nn.softmax(logits, axis=-1)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = squash(preactivate)
        activations = activations.write(i, activation)

        act_3d = tf.expand_dims(activation, 1)
        act_replicated = tf.tile(act_3d, [1, input_dim, 1, 1, 1, 1])
        distances = tf.reduce_sum(votes * act_replicated, axis=-1)
        logits += distances
        return i + 1, logits, activations

    activations = tf.TensorArray(
      dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
      lambda i, logits, activations: i < num_routing,
      _body,
      loop_vars=[i, logits, activations],
      swap_memory=True)

    return tf.cast(activations.read(num_routing - 1), dtype='float32')
