import tensorflow as tf


class Generator(tf.keras.Model):
    """
    2nd generation generator
    """
    def __init__(self, config, prefix='Generator'):
        self.config = config
        self.prefix = prefix

    def call(self, inputs, training=None, mask=None):
        pass

    def build_model(self, input_tensor):
        out_tensor = self.call(input_tensor)
        return tf.keras.Model(input_tensor, out_tensor)


class Discriminator(tf.keras.Model):
    """
    2nd generation discriminator
    """
    def __init__(self, config, prefix='Discriminator'):
        self.config = config
        self.prefix = prefix

    def call(self, inputs, training=None, mask=None):
        pass

    def build_model(self, input_tensor):
        out_tensor = self.call(input_tensor)
        return tf.keras.Model(input_tensor, out_tensor)

