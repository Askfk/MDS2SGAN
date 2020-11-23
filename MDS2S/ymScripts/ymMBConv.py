import tensorflow as tf

import ymLayers
from config import Config


class MBConv(tf.keras.layers.Layer):
    """Mobile Inverted Residual Bottleneck."""

    def __init__(self, block_args, activation, drop_rate=None, prefix='', **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.block_args = block_args
        self.activation = activation
        self.drop_rate = drop_rate
        self.prefix = prefix

    def get_config(self):
        config = super().get_config()
        config.update(self.block_args)
        config['activation'] = self.activation
        config['drop_rate'] = self.drop_rate
        config['prefix'] = self.prefix
        return config

    def call(self, inputs, **kwargs):
        has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        Dropout = ymLayers.FixedDropout
        filters = self.block_args.input_filters * self.block_args.expand_ratio

        if self.block_args.expand_ratio != 1:
            x = tf.keras.layers.Conv2D(filters, 1,
                                       padding='same',
                                       use_bias=False,
                                       kernel_initializer=Config.CONV_KERNEL_INITIALIZER,
                                       name=self.prefix + 'expand_conv')(inputs)
            x = ymLayers.BatchNorm(name=self.prefix + 'expand_bn')(x, training=Config.TRAIN_BN)
            x = tf.keras.layers.Activation(self.activation, name=self.prefix + 'expand_activation')(x)

        else:
            x = inputs

        # Depthwise Convolution
        x = tf.keras.layers.DepthwiseConv2D(self.block_args.kernel_size,
                                            strides=self.block_args.strides,
                                            padding='same',
                                            use_bias=False,
                                            depthwise_initializer=Config.CONV_KERNEL_INITIALIZER,
                                            name=self.prefix + 'dwconv')(x)
        x = ymLayers.BatchNorm(name=self.prefix + 'bn')(x, training=Config.TRAIN_BN)
        x = tf.keras.layers.Activation(self.activation, name=self.prefix + 'activation')(x)

        if has_se:
            num_reduced_filters = max(1, int(
                self.block_args.input_filters * self.block_args.se_ratio
            ))
            se_tensor = tf.keras.layers.GlobalAveragePooling2D(name=self.prefix + 'se_squeeze')(x)
            target_shape = (1, 1, filters) if tf.keras.backend.image_data_format() == 'channels_last' else (
                filters, 1, 1)
            se_tensor = tf.keras.layers.Reshape(target_shape, name=self.prefix + 'se_reshape')(se_tensor)
            se_tensor = tf.keras.layers.Conv2D(num_reduced_filters, 1,
                                               activation=self.activation,
                                               padding='same',
                                               use_bias=True,
                                               kernel_initializer=Config.CONV_KERNEL_INITIALIZER,
                                               name=self.prefix + 'se_reduce')(se_tensor)
            se_tensor = tf.keras.layers.Conv2D(filters, 1,
                                               activation='sigmoid',
                                               padding='same',
                                               use_bias=True,
                                               kernel_initializer=Config.CONV_KERNEL_INITIALIZER,
                                               name=self.prefix + 'se_expand')(se_tensor)
            x = tf.keras.layers.multiply([x, se_tensor], name=self.prefix + 'se_excite')

        # Output phase
        x = tf.keras.layers.Conv2D(self.block_args.output_filters, 1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=Config.CONV_KERNEL_INITIALIZER,
                                   name=self.prefix + 'project_conv')(x)
        x = ymLayers.BatchNorm(name=self.prefix + 'project_bn')(x, training=Config.TRAIN_BN)

        if self.block_args.id_skip and all(
                s == 1 for s in self.block_args.strides
        ) and self.block_args.input_filters == self.block_args.output_filters:
            if self.drop_rate and (self.drop_rate > 0):
                x = Dropout(self.drop_rate,
                            noise_shape=(None, 1, 1, 1),
                            name=self.prefix + 'drop')(x)
            x = tf.keras.layers.add([x, inputs], name=self.prefix + 'add')

        return x

