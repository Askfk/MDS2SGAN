import tensorflow as tf

from ymLayers import BatchNorm, FixedDropout, DilatedConv2d
from ymActivations import swish, relu, leakyRelu
from BackBones.backbone import build_backbone_net_graph


def customStrides(n):
    if n <= 0:
        return 1
    if n >= 3:
        return 2
    return n


class Encoder(tf.keras.Model):
    """

    input shape: [batch, 96， 96， 3]
    """

    def __init__(self, config, prefix='encoder', **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.config = config
        self.repeat_times = config.ENCODER_REPEAT
        self.prefix = prefix

        # Output shape [batch, config.SIGNAL_FREQ / 2, config.SIGNAL_PERIOD / 2, 32]
        self.conv = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu',
                                           strides=1, kernel_regularizer='l1_l2',
                                           name='encoder_conv2d_layer')
        self.ac = tf.keras.layers.Activation(relu)
        self.bn = BatchNorm()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.ac(x)
        x = self.bn(x, training=training)
        n = x.shape[-1]
        out = []
        for i in range(self.repeat_times):
            # x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same',
            #                                     depth_multiplier=2, depthwise_regularizer='l1_l2',
            #                                     name=self.prefix + 'depthwiseconv{}'.format(i + 1))(x)
            x = tf.keras.layers.Conv2D(n * 2**(i + 1), (3, 3), strides=customStrides(i), padding='same',
                                       name=self.prefix + 'repeat_conv{}'.format(i + 1))(x)
            x = tf.keras.layers.Activation(swish, name=self.prefix + 'dilated{}_ac'.format(i + 1))(x)
            x = BatchNorm()(x, training=training)
            out.append(x)
            # x = FixedDropout(0.3 / (i + 1), noise_shape=(None, 1, 1, 1),
            #                  name=self.prefix + 'dropout{}'.format(i + 1))(x)

        return out

    def build_model(self, input_tensor):
        outputs = self.call(input_tensor)
        return tf.keras.Model(input_tensor, outputs, name='MDS_encoder')


class Decoder(tf.keras.Model):
    """
    input shape: [batch, 108, 108, 256]
    output shape: [batch, config.SIGNAL_FREQ, config.SIGNAL_PERIOD, config.NUM_STATUS]
    """

    def __init__(self, config, prefix='decoder', **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.config = config
        self.prefix = prefix

        # TODO: should reconsider the activations.

        # out: [batch, 24, 24, 256]
        self.in_layer = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), activation='relu',
                                               padding='same', kernel_regularizer='l1_l2', name=self.prefix+'_in_layer')
        # out: [batch, 24, 24, 128]
        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), (1, 1), padding='same',
                                                       activation='relu', kernel_regularizer='l1_l2',
                                                       name=self.prefix+'_deconv1')
        self.bn1 = BatchNorm()
        # out: [batch, 48, 48, 32]
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64, (5, 5), (2, 2), padding='same', activation='relu',
                                                       kernel_regularizer='l1_l2', name=self.prefix+'_deconv2')
        self.bn2 = BatchNorm()
        # out: [batch, 96, 96, config.NUM_STATUS]
        self.deconv3 = tf.keras.layers.Conv2DTranspose(config.NUM_MODALS * 3, (5, 5), (2, 2), padding='same',
                                                       use_bias=False, activation='tanh', name=self.prefix+'_deconv3')

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
def get_encoders_graph(config, input_tensor=None):
    if config.ENCODER_BACKBONE == 'custom':
        encoder = Encoder(config, prefix='encoder')
        if input_tensor:
            return encoder.build_model(input_tensor)
        return encoder
    else:
        return build_backbone_net_graph(config.ENCODER_BACKBONE, config)


# build decoder backbones
def get_decoders_graph(config, input_tensor=None):
    if config.DECODER_BACKBONE == 'custom':
        decoder = Decoder(config, prefix='decoder')
        if input_tensor:
            return decoder.build_model(input_tensor)
        return decoder
    else:
        return build_backbone_net_graph(config.DECODER_BACKBONE, config)


if __name__ == '__main__':
    from config import Config

    input_t = tf.keras.layers.Input([96, 96, 3])

    config = Config()
    encoder = get_encoders_graph(config)
    out = encoder(input_t)
    for o in out:
        print(o.shape)


