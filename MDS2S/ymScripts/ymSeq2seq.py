import tensorflow as tf
import numpy as np

from .ymLayers import BatchNorm, FixedDropout, DilatedConv2d
from .ymActivations import swish, relu, leakyRelu
from ..BackBones.backbone import build_backbone_net_graph


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

    def __init__(self, config, prefix='encoder_cnn', **kwargs):
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
        return tf.keras.Model(input_tensor, outputs, name=self.prefix)


class Middle(tf.keras.Model):
    """
    Middle part between Encoder and Decoder
    """

    def __init__(self, config, prefix='Middle_cnn', **kwargs):
        super(Middle, self).__init__(**kwargs)

        self.config = config
        self.prefix = prefix

        self.middle_1 = tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (5, 5), padding='valid',
                                               name=self.prefix+'_middle_1', activation='relu')
        self.bn1 = BatchNorm()
        self.middle_2 = tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (5, 5), padding='valid',
                                               name=self.prefix+'_middle_2', activation='relu')
        self.bn2 = BatchNorm()
        self.middle_3 = tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (5, 5), padding='valid',
                                               name=self.prefix+'_middle_3', activation='relu')
        self.bn3 = BatchNorm()

    def call(self, inputs, training=False):
        x = self.concatPyramidFeatures(features=inputs)
        x = self.middle_1(x)
        x = self.bn1(x, training=training)
        x = self.middle_2(x)
        x = self.bn2(x, training=False)
        x = self.middle_3(x)
        x = self.bn3(x, training=False)

        for i in range(2):
            x = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), strides=2, padding='same',
                                       name=self.prefix+'_downsamping_{}'.format(i+1), activation='relu')(x)
            x = BatchNorm()(x, training=training)
        return x

    def concatPyramidFeatures(self, features):
        for i in range(len(features)):
            features[i] = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same',
                                                   activation='relu')(features[i])
        [_, o1, o2, _, o3] = features

        # TODO: find out whether need to set a middle layer between encoder and decoder. The task
        #  of middle layer is to combine the pyramid feature maps
        o1 = tf.reshape(o1, [-1, o1.shape[1] ** 2, o1.shape[3]], name=self.prefix + 'o1_reshape')
        o2 = tf.reshape(o2, [-1, o2.shape[1] ** 2, o2.shape[3]], name=self.prefix + 'o2_reshape')
        o3 = tf.reshape(o3, [-1, o3.shape[1] ** 2, o3.shape[3]], name=self.prefix + 'o3_reshape')

        x = tf.concat([o1, o2, o3], axis=1, name=self.prefix + 'o_concat')
        x = tf.reshape(x, [-1, self.config.DECODER_INPUT_SHAPE[1], self.config.DECODER_INPUT_SHAPE[1], x.shape[2]],
                       name=self.prefix + 'x_reshape')
        # x.set_shape([-1, self.config.DECODER_INPUT_SHAPE[1], self.config.DECODER_INPUT_SHAPE[1], x.shape[2]])
        return x

    def build_model(self, input_tensor):
        out = self.call(input_tensor)
        return tf.keras.Model(input_tensor, out, name=self.prefix)


class Decoder(tf.keras.Model):
    """
    input shape: [batch, 108, 108, 256]
    output shape: [batch, config.SIGNAL_FREQ, config.SIGNAL_PERIOD, config.NUM_STATUS]
    """

    def __init__(self, config, prefix='decoder_cnn', **kwargs):
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
        return tf.keras.Model(input_tensors, outputs, name=self.prefix)


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

    input_t = tf.random.uniform([6, 96, 96, 256])
    #
    config = Config()
    encoder = get_encoders_graph(config)
    out = encoder(input_t)
    for o in out:
        print(o.shape)
    middle = Middle(config)
    out = middle(out, True)
    print(out.shape)


