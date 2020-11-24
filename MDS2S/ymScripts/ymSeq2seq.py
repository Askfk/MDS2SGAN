import tensorflow as tf

from .ymLayers import BatchNorm, FixedDropout
from .ymActivations import swish, relu
from ..BackBones.backbone import build_backbone_net_graph


class Encoder(tf.keras.Model):
    """

    input shape: [batch, config.SIGNAL_FREQ, config.SIGNAL_PERIOD, 1]
    """

    def __init__(self, repeat_times, blocks_args, config, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.config = config
        self.repeat_times = repeat_times
        self.blocks_args = blocks_args

        # Output shape [batch, config.SIGNAL_FREQ / 2, config.SIGNAL_PERIOD / 2, 32]
        self.conv = tf.keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu',
                                           strides=1,
                                           name='encoder_conv2d_layer')
        self.ac = tf.keras.layers.Activation(relu)
        self.bn = BatchNorm()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.ac(x)
        x = self.bn(x, training=training)

        for i in range(self.repeat_times):
            # TODO: Add mvconv blocks or depthwiseconv layers
            pass

        return x

    def build_model(self, input_tensor):
        outputs = self.call(input_tensor)
        return tf.keras.Model(input_tensor, outputs, name='MDS_encoder')


class Decoder(tf.keras.Model):
    """
    input shape: [batch, 36, 36, 256]
    output shape: [batch, config.SIGNAL_FREQ, config.SIGNAL_PERIOD, config.NUM_STATUS]
    """

    def __init__(self, block_args, config, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.config = config
        self.block_args = block_args

        # TODO: should reconsider the activations.

        # out: [batch, 24, 24, 256]
        self.in_layer = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (7, 7), activation='relu')
        # out: [batch, 24, 24, 128]
        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), (1, 1), padding='same',
                                                       activation='relu')
        self.bn1 = BatchNorm()
        # out: [batch, 48, 48, 32]
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64, (5, 5), (2, 2), padding='same', activation='relu')
        self.bn2 = BatchNorm()
        # out: [batch, 96, 96, config.NUM_STATUS]
        self.deconv3 = tf.keras.layers.Conv2DTranspose(config.NUM_MODALS * 3, (5, 5), (2, 2), padding='same',
                                                       use_bias=False, activation='tanh')

    def call(self, inputs, training=False):

        x = self.in_layer(inputs)
        x = self.deconv1(x)
        x = self.bn1(x, training=training)
        x = self.deconv2(x)
        x = self.bn2(x, training=training)
        x = self.deconv3(x)

        return x

    def build_model(self, input_tensors):
        outputs = self.call(input_tensors)
        return tf.keras.Model(input_tensors, outputs, name='MDS_decoder')


# build encoder backbones
def get_encoders_graph(config, input_tensor):
    if config.ENCODER_BACKBONE == 'custom':
        encoder = Encoder(config.DEFAULT_BLOCKS_ARGS, config)
        return encoder.build_model(input_tensor)
    else:
        return build_backbone_net_graph(config.ENCODER_BACKBONE, config)


# build decoder backbones
def get_decoders_graph(config, input_tensor):
    if config.DECODER_BACKBONE == 'custom':
        decoder = Decoder(config.DEFAULT_BLOCKS_ARGS, config)
        return decoder
    else:
        return build_backbone_net_graph(config.DECODER_BACKBONE, config)


if __name__ == '__main__':
    from config import Config
    import numpy as np

    config = Config()
    encoder = get_encoders_graph(config, None)
    x = tf.convert_to_tensor(np.random.random([1, 128, 128, 3]), tf.float32)
    out = encoder(x, False)
    for o in out:
        print(o.shape)
