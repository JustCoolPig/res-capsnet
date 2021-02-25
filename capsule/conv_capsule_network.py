

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
from tensorflow.keras.layers import BatchNormalization


class ConvCapsNet(tf.keras.Model):

    def __init__(self, args):
        super(ConvCapsNet, self).__init__()

        # Set params
        dimensions = list(map(int, args.dimensions.split(","))) if args.dimensions != "" else []
        layers = list(map(int, args.layers.split(","))) if args.layers != "" else []

        self.use_bias=args.use_bias
        self.use_reconstruction = args.use_reconstruction
        self.make_skips = args.make_skips
        self.skip_dist = args.skip_dist

        if args.dataset == 'mnist':
            img_size = 24 
        elif args.dataset == 'cifar10':
            img_size = 32
        else:
            raise NotImplementedError()
        conv1_filters, conv1_kernel, conv1_stride = 128, 7, 2
        out_height = (img_size - conv1_kernel) // conv1_stride + 1
        out_width = (img_size - conv1_kernel) // conv1_stride + 1 

        with tf.name_scope(self.name):

            # normal convolution
            self.conv_1 = tf.keras.layers.Conv2D(
                        conv1_filters, 
                        kernel_size=conv1_kernel, 
                        strides=conv1_stride, 
                        padding='valid', 
                        activation="relu", 
                        name="conv1")

            # reshape into capsule shape
            self.capsuleShape = tf.keras.layers.Reshape(target_shape=(out_height, out_width, 1, conv1_filters), name='toCapsuleShape')

            self.batchNorm = BatchNormalization()
            
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
                            kernel_size=3,
                            routing_iterations=args.iterations))

            # flatten for input to FC capsule
            self.flatten = tf.keras.layers.Reshape(target_shape=(out_height * out_width * layers[-2], dimensions[-2]), name='flatten')
            
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
            self.residual = Residual()


    # Inference
    def call(self, x, y):
        x = self.conv_1(x)
        x = self.batchNorm(x)
        x = self.capsuleShape(x)
        layers = [x]

        capsule_outputs = []    
        i = 0    
        for j, capsule in enumerate(self.capsule_layers):
            x = capsule(x)
            #print('shape: ', x.shape)
            
            # add skip connection
            capsule_outputs.append(x)
            if self.make_skips and i > 0 and i % self.skip_dist == 0:
                out_skip = capsule_outputs[j-self.skip_dist]
                if x.shape == out_skip.shape:
                    #print('make residual connection from ', j-self.skip_dist, ' to ', j)
                    x = self.residual(x, out_skip)
                    i = -1
            
            i += 1
            layers.append(x)

        x = self.flatten(x)
        x = self.FCCapsuleLayer(x)
 
        r = self.reconstruction_network(x, y) if self.use_reconstruction else None
        out = self.norm(x)

        return out, r, layers
