import tensorflow as tf
import numpy as np


class BatchNorm(tf.keras.layers.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
        to make changes if needed.

        Batch normalization has a negative effect on training if batches are small
        so this layer is often frozen (via setting in Config class) and functions
        as linear layer.
        """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = tf.keras.backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


class DilatedConv2d(tf.keras.layers.Layer):
    """
    This is a kind of discrete convolution layer
    Defaulted set activation as relu.
    """

    def __init__(self, out_c, kernel_size=3, padding='SAME', strides=1, dilations=1, **kwargs):
        super(DilatedConv2d, self).__init__(**kwargs)
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilations = dilations

    def get_config(self):
        config = super().get_config()
        config['out_c'] = self.out_c
        config['kernel_size'] = self.kernel_size
        config['padding'] = self.padding
        config['strides'] = self.strides
        config['dilations'] = self.dilations
        return config

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                      shape=[input_shape[-1], self.kernel_size, self.kernel_size, self.out_c],
                                      dtype=tf.float32)

    def call(self, inputs):
        """The size of input must be [N, H, W, C]"""
        x = inputs
        out = tf.nn.dilation2d(x, self.kernel, strides=[1, 1, 1, 1], padding=self.padding, dilations=self.dilations,
                               data_format="NHWC")
        out = tf.nn.relu(out)
        return out


class DepthwiseConv2d(tf.keras.layers.Layer):
    """
    This is a customized depth-wise convolutional layer.
    Activation defaulted set relu
    """

    def __init__(self, scale, kernel_size=3, padding='SAME', strides=1, dilations=None, **kwargs):
        super(DepthwiseConv2d, self).__init__(**kwargs)
        self.scale = scale
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilations = dilations

    def get_config(self):
        config = super().get_config()
        config['scale'] = self.scale
        config['kernel_size'] = self.kernel_size
        config['padding'] = self.padding
        config['strides'] = self.strides
        config['dilations'] = self.dilations
        return config

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                      shape=[self.kernel_size, self.kernel_size, input_shape[-1], self.scale],
                                      dtype=tf.float32)

    def call(self, inputs):
        """The size of input must be [N, H, W, C]"""
        x = inputs
        out = tf.nn.depthwise_conv2d(x, self.kernel, [1, self.strides, self.strides, 1],
                                     self.padding, dilations=self.dilations)
        out = tf.nn.relu(out)
        return out



