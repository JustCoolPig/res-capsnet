

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
        conv1_filters, conv1_kernel, conv1_stride = 128, 9, 2
        out_height = math.ceil((img_size - conv1_kernel) / conv1_stride) + 1
        out_width = math.ceil((img_size - conv1_kernel) / conv1_stride) + 1

        with tf.name_scope(self.name):

            # normal convolution
            self.conv_1 = tf.keras.layers.Conv2D(
                        conv1_filters, 
                        kernel_size=conv1_kernel, 
                        strides=conv1_stride, 
                        padding='same', 
                        activation="relu", 
                        name="conv1")

            # reshape into capsule shape
            self.capsuleShape = tf.keras.layers.Reshape(target_shape=(out_height, out_width, 1, conv1_filters), name='toCapsuleShape')
            
            # convolutional capsule layers
            self.capsule_layers = []
            for i in range(len(layers)-1):
                self.capsule_layers.append(
                    ConvCapsule(
                            name="ConvCapsuleLayer" + str(i), 
                            in_capsules=layers[i], 
                            in_dim=dimensions[i], 
                            out_dim=dimensions[i], 
                            out_capsules=layers[i+1], 
                            kernel_size=7))

            # flatten for input to FC capsule
            self.flatten = tf.keras.layers.Reshape(target_shape=(out_height * out_width * layers[-2], dimensions[0]), name='flatten')
            
            # fully connected caspule layer
            self.FCCapsuleLayer = Capsule(
                        name="FCCapsuleLayer",
                        in_capsules = out_height * out_width * layers[-2], 
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
        x = self.capsuleShape(x)

        for caps_layer in self.capsule_layers:
            x = caps_layer(x)

        x = self.flatten(x)
        x = self.FCCapsuleLayer(x)
 
        r = self.reconstruction_network(x, y) if self.use_reconstruction else None
        out = self.norm(x)

        return out, r, [x]
