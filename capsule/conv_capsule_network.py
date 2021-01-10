

import tensorflow as tf
import math

from capsule.capsule_layer import Capsule
from capsule.em_capsule_layer import EMCapsule
from capsule.gamma_capsule_layer import GammaCapsule
from capsule.conv_capsule_layer import ConvCapsule
from capsule.primary_capsule_layer import PrimaryCapsule
from capsule.reconstruction_network import ReconstructionNetwork
from capsule.norm_layer import Norm
from capsule.residual_layer import Residual


class ConvCapsNet(tf.keras.Model):

    def __init__(self, args):
        super(ConvCapsNet, self).__init__()

        # Set params
        dimensions = list(map(int, args.dimensions.split(","))) if args.dimensions != "" else []
        layers = list(map(int, args.layers.split(","))) if args.layers != "" else []

        self.use_bias=args.use_bias
        self.use_reconstruction = args.use_reconstruction

        img_size = 30
        conv1_filters = 128
        conv1_kernel = 9
        conv1_stride = 2
        out_height = math.ceil((img_size - conv1_kernel) / conv1_stride) + 1
        out_width = math.ceil((img_size - conv1_kernel) / conv1_stride) + 1

        with tf.name_scope(self.name):

            self.conv_1 = tf.keras.layers.Conv2D(conv1_filters, kernel_size=conv1_kernel, strides=conv1_stride, padding='same', activation="relu", name="conv1")

            self.capsuleShape = tf.keras.layers.Reshape(target_shape=(out_height, out_width, 1, conv1_filters))
            
            self.primary = ConvCapsule(
                        name="PrimaryConvCapsuleLayer", 
                        in_capsules=layers[0], 
                        in_dim=dimensions[0], 
                        out_dim=dimensions[0], 
                        out_capsules=64, 
                        kernel_size=7)

            self.flatten = tf.keras.layers.Reshape(target_shape=(out_height * out_width * layers[0], dimensions[0]))
            
            self.FCCapsuleLayer = Capsule(
                        name="FCCapsuleLayer",
                        in_capsules = out_height * out_width * layers[0], 
                        in_dim = dimensions[-2], 
                        out_capsules = layers[-1],
                        out_dim = dimensions[-1], 
                        use_bias = self.use_bias)                    

            if self.use_reconstruction:
                self.reconstruction_network = ReconstructionNetwork(
                    name="ReconstructionNetwork",
                    in_capsules=layers[-1], 
                    in_dim=dimensions[-1],
                    out_dim=args.img_height,
                    img_dim=args.img_depth)

            self.norm = Norm()


    # Inference
    def call(self, x, y):

        x = self.conv_1(x)
        print('conv1', x.shape)
        x = self.capsuleShape(x)

        print('caps', x.shape)
        x = self.primary(x)

        print('primary', x.shape)
        x = self.flatten(x)
        print('flatten', x.shape)
        x = self.FCCapsuleLayer(x)
 
        r = self.reconstruction_network(x, y) if self.use_reconstruction else None
        out = self.norm(x)

        return out, r, [x]
