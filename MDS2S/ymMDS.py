import tensorflow as tf

from .ymScripts.ymActivations import swish, leakyRelu
from .ymScripts.ymLayers import BatchNorm, FixedDropout
from .ymScripts.ymSeq2seq import get_encoders_graph, get_decoders_graph, Middle


tf.config.experimental_run_functions_eagerly(True)


class MDS(tf.keras.Model):
    """Mode Decompose Seq2seq Model
    Seq2seq architecture with encoder and decoder as generator"""

    # TODO: re-design the architecture of this generator

    def __init__(self, config, prefix='', **kwargs):
        super(MDS, self).__init__(**kwargs)
        self.config = config
        self.prefix = prefix

        self.encoder = get_encoders_graph(config, None)

        self.middle = Middle(config)

        self.decoder = get_decoders_graph(config, None)

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training)
        # x = self.middle(x, training)
        x = self.backup(x)
        x = self.decoder(x, training)

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

    def backup(self, features):
        [_, o1, o2, o3, _] = features
        return o3

    def build_model(self, inp_tensor):
        out_tensor = self.call(inp_tensor, False)
        return tf.keras.Model(inp_tensor, out_tensor, name='MDS')


class DCM(tf.keras.Model):
    """Discriminator Classifier Model"""

    def __init__(self, config, prefix='', **kwargs):
        super(DCM, self).__init__(**kwargs)
        self.prefix = prefix + '_'
        self.repeat_times = config.DCM_REPEAT
        self.config = config

        # Output: [batch, 2000, 2000, 16]
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer='l1_l2', name=self.prefix + 'conv1')
        self.activation1 = tf.keras.layers.Activation(leakyRelu)
        self.bn1 = BatchNorm()

        # # output: [batch, 1000, 1000, 32]
        # self.depthwiseconv2 = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2)
        # self.ac2 = tf.keras.layers.Activation(swish, name=self.prefix + 'dpwconv2_ac')
        # self.bn2 = BatchNorm()
        # self.dp2 = FixedDropout(0.3, noise_shape=(None, 1, 1, 1), name=self.prefix + 'drop2')
        #
        # # output: [batch, 500, 500, 64]
        # self.depthwiseconv3 = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2)
        # self.ac2 = tf.keras.layers.Activation(swish, name=self.prefix + 'dpwconv2_ac')
        # self.bn2 = BatchNorm()
        # self.dp2 = FixedDropout(0.3, noise_shape=(None, 1, 1, 1), name=self.prefix + 'drop2')

        self.final_dense = tf.keras.layers.Dense(self.config.NUM_CLASSES)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        """The shape of inputs  is [batch, config.SIGNAL_FREQ, config.SIGNAL_PERIOD, config.NUM_STATUS]"""

        # output: [batch, SIGNAL_FREQ / 2, SIGNAL_PERIOD / 2, 16]
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.bn1(x, training=training)

        # in every repeat, resolution decrease 4 times, channel increase 2 times
        for i in range(self.repeat_times):
            x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same',
                                                depth_multiplier=2, depthwise_regularizer='l1_l2',
                                                name=self.prefix + 'depthwiseconv{}'.format(i + 1))(x)
            x = tf.keras.layers.Activation(leakyRelu, name=self.prefix + 'dpwconv{}_ac'.format(i + 1))(x)
            x = BatchNorm()(x, training=training)
            # x = FixedDropout(0.3 / (i + 1), noise_shape=(None, 1, 1, 1),
            #                  name=self.prefix + 'dropout{}'.format(i + 1))(x)

        # TODO: Find out whether need to squeeze the feature maps before passing them into dense layer.

        # Flatten x,
        x = tf.keras.layers.Flatten()(x)
        logits = self.final_dense(x)
        out = self.softmax(logits)
        return logits, out

    def build_model(self, input_tensors):
        outputs = self.call(input_tensors, self.config.TRAIN_BN)
        return tf.keras.Model(input_tensors, outputs, name=self.prefix)


# TODO: Find a way to make input can only have 1 channel rather than must be exact 3 channels


if __name__ == '__main__':
    import numpy as np
    from config import Config
    import matplotlib.pyplot as plt
    config = Config()
    mds = MDS(config)
    dcm = DCM(config)

    # inp = tf.convert_to_tensor(np.random.random([3, 128, 128, 3]), tf.float32)
    # out = mds(inp, False)
    # inp = tf.keras.layers.Input([96, 96, 3])
    inp = tf.random.uniform([6, 96, 96, 3])
    mds_out = mds(inp, False)
    print(mds_out.shape)
    logits, dcm_out = dcm(mds_out, False)
    # model = tf.keras.Model(inp, dcm_out)
    # model.summary()
    print(logits.shape)
    print(dcm_out.shape)

