import tensorflow as tf
import numpy as np


class LSTMs(tf.keras.Model):
    """Use LSTN as encoder or decoder."""

    def __init__(self, **kwargs):
        super(LSTMs, self).__init__(**kwargs)

