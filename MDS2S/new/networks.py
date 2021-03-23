import tensorflow as tf
import numpy as np
from ymScripts.ymLayers import BatchNorm


class Generator(tf.keras.Model):
    """
    Generator
    """

    def __init__(self, config, repeat=3, prefix='Generator', **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.config = config
        self.repeat = repeat
        self.prefix = prefix

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same',
                                            name=self.prefix+"_conv1")
        self.bn1 = BatchNorm()
        self.ac1 = tf.keras.activations.swish

        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same',
                                            name=self.prefix + "_conv2")
        self.bn2 = BatchNorm()
        self.ac2 = tf.keras.activations.swish

        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=2, padding='same',
                                            name=self.prefix + "_conv3")
        self.bn3 = BatchNorm()
        self.ac3 = tf.keras.activations.swish

        self.conv4 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=2, padding='same',
                                            name=self.prefix + "_conv4")
        self.bn4 = BatchNorm()
        self.ac4 = tf.keras.activations.swish

        self.conv = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', name=self.prefix+"_final_conv")
        self.bn = BatchNorm()
        self.ac = tf.keras.activations.swish

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)  # 72
        x = self.bn1(x, training=training)
        x = self.ac1(x)

        x = self.conv2(x)  # 36 128
        x = self.bn2(x, training=training)
        x = self.ac2(x)

        x1 = x

        x = self.conv3(x)  # 18 256
        x = self.bn3(x, training=training)
        x = self.ac3(x)

        x2 = x

        x = self.conv4(x)  # 9 256
        x = self.bn4(x, training=training)
        x = self.ac4(x)

        x3 = x

        x1 = tf.keras.layers.Conv2D(128, (1, 1), strides=2, padding='same')(x1)
        x2 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(x2)

        x2 = tf.concat([x1, x2], axis=-1)

        x2 = tf.keras.layers.Conv2D(256, (1, 1), strides=2, padding='same')(x2)
        x3 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(x3)

        feats = tf.concat([x2, x3], axis=-1)

        x = feats
        for i in range(self.repeat):
            x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name=self.prefix+'_IMFs_3x3conv{}'.format(i+1))(x)
            x = BatchNorm()(x, training=training)
            x = tf.keras.activations.swish(x)

            x = tf.keras.layers.Conv2D(256, (1, 1), padding='same', name=self.prefix+'_IMFs_1x1conv{}'.format(i+1))(x)
            x = BatchNorm()(x, training=training)
            x = tf.keras.activations.swish(x)
        x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, padding='valid')(x)
        x = BatchNorm()(x, training=training)
        x = tf.keras.activations.swish(x)  # 11x11x128

        x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, padding='valid')(x)
        x = BatchNorm()(x, training=training)
        x = tf.keras.activations.swish(x)  # 13x13x64

        x = tf.keras.layers.Conv2DTranspose(36, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = BatchNorm()(x, training=training)
        x = tf.keras.activations.swish(x)  # 26x26x36

        imfs = tf.reshape(x, [-1, 2704, 9])

        return feats, imfs

    def build_model(self, input_tensor):
        out_tensor = self.call(input_tensor)
        return tf.keras.Model(input_tensor, out_tensor, name=self.prefix)


class Discriminator(tf.keras.Model):
    """
    Discriminator
    """
    def __init__(self, config, repeat=3, prefix='Discriminator', **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.config = config
        self.repeat = repeat
        self.prefix = prefix

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(self.repeat):
            x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name=self.prefix+"_3x3conv{}".format(i+1))(x)
            x = BatchNorm()(x, training=training)
            x = tf.keras.activations.swish(x)
            x = tf.keras.layers.Conv2D(256, (1, 1), padding='same', name=self.prefix+"_1x1conv{}".format(i+1))(x)
            x = BatchNorm()(x, training=training)
            x = tf.keras.activations.swish(x)

        pm = tf.keras.layers.Conv2D(4, (1, 1), padding='same', name=self.prefix+"_possi")(x)
        pm = BatchNorm()(pm, training=training)
        pm = tf.keras.activations.swish(pm)
        pm = tf.keras.layers.Flatten()(pm)
        pm = tf.keras.layers.Dense(2)(pm)
        pm_logits = tf.reshape(pm, [-1, 2])
        pm = tf.nn.softmax(pm_logits, axis=-1)

        dm = tf.keras.layers.Conv2D(4, (1, 1), padding='same', name=self.prefix + "_depth")(x)
        dm = BatchNorm()(dm, training=training)
        dm = tf.keras.activations.swish(dm)
        dm = tf.keras.layers.Flatten()(dm)
        dm = tf.keras.layers.Dense(1)(dm)
        dm = tf.reshape(dm, [-1, 1])

        lm = tf.keras.layers.Conv2D(128, (1, 1), padding='same', name=self.prefix+"_loc")(x)
        lm = BatchNorm()(lm, training=training)
        lm = tf.keras.activations.swish(lm)
        lm = tf.reshape(lm, [-1, 144, 72])

        return pm, dm, lm, pm_logits

    def build_model(self, input_tensor):
        out_tensor = self.call(input_tensor)
        return tf.keras.Model(input_tensor, out_tensor, name=self.prefix)


if __name__ == '__main__':
    from config import Config
    from DataGenerator import DataGenerator

    config = Config()
    generator = DataGenerator(config).generator
    print(">>>>>>>>>>>>>>Dataset Initialization Done")
    dataset = tf.data.Dataset.from_generator(generator,
                                             (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                             (tf.TensorShape([144, 144, 57]),
                                              tf.TensorShape([2800, 9]),
                                              tf.TensorShape([144, 72]),
                                              tf.TensorShape([2]),
                                              tf.TensorShape([None])))
    dataset = dataset.batch(8).shuffle(16)

    g = Generator(config)
    d = Discriminator(config)

    for input_tensor, _, loc, damage_matrix, depth in dataset.take(1):
        feats, imfs = g(input_tensor)
        print(feats.shape, imfs.shape)

        pm, dm, lm, _ = d(feats)
        print(pm.shape, dm.shape, lm.shape)
