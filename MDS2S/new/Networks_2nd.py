import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization as BatchNorm

class Generator(tf.keras.Model):
    """
    2nd generation generator
    """
    def __init__(self, config, prefix='Generator', **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.config = config
        self.prefix = prefix

    def call(self, inputs, training=None, mask=None):
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name=self.prefix+"_in_conv")(inputs)
        x = BatchNorm()(x, training=True)
        x = tf.keras.activations.swish(x)

        for i in range(4):
            x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', depth_multiplier=2)(x)
            x = BatchNorm()(x, training=True)
            x = tf.keras.activations.swish(x)
        x = tf.keras.layers.DepthwiseConv2D((1, 1), strides=(1, 1), padding='same', depth_multiplier=1)(x)
        x = BatchNorm()(x, training=training)
        x = tf.keras.activations.swish(x)

        # feats : [N, 9, 9, 512]
        feats = x

        x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, padding='valid',
                                            name=self.prefix+"_convT_1")(x)
        x = BatchNorm()(x, training=training)
        x = tf.keras.activations.swish(x)  # 11x11x128

        x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, padding='valid',
                                            name=self.prefix+"_convT_2")(x)
        x = BatchNorm()(x, training=training)
        x = tf.keras.activations.swish(x)  # 13x13x64

        x = tf.keras.layers.Conv2DTranspose(36, kernel_size=(3, 3), strides=2, padding='same',
                                            name=self.prefix+"_convT_3")(x)
        x = BatchNorm()(x, training=training)
        x = tf.keras.activations.swish(x)  # 26x26x36

        imfs = tf.reshape(x, [-1, 2704, 9])
        return feats, imfs

    def build_model(self, input_tensor):
        out_tensor = self.call(input_tensor)
        return tf.keras.Model(input_tensor, out_tensor)


if __name__ == '__main__':
    from config import Config
    from Networks_1st import Discriminator
    cfg = Config()
    input_tensor = tf.keras.Input([144, 144, 57])
    g = Generator(cfg)
    d = Discriminator(cfg)
    feats, imfs = g(input_tensor)
    print(feats.shape, imfs.shape)
    pm, dm, lm, _ = d(feats)
    print(pm.shape, dm.shape, lm.shape)
