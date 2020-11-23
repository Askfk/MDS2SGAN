import tensorflow as tf
import ymLayers as yml


class CustomModel(tf.keras.Model):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.l1 = yml.DilatedConv2d(32, name='l1')
        self.l2 = yml.DepthwiseConv2d(64, name='l2')
        self.l3 = tf.keras.layers.Conv2D(128, 3, name='l3')

    def call(self, inputs):
        x = inputs
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        return out


l1 = yml.DilatedConv2d(32, strides=2, padding='SAME', name='l1')
l2 = yml.DepthwiseConv2d(2, strides=2, padding='SAME', name='l2')
l3 = tf.keras.layers.Conv2D(128, 3, strides=2, padding='SAME', name='l3')
input = tf.keras.layers.Input([1024, 1024, 3])
out = l1(input)
out = l2(out)
out = l3(out)
mm = tf.keras.Model(input, out)
mm.summary()