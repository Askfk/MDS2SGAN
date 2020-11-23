import tensorflow as tf


def swish(x):
    """Swish activation function: x * sigmoid(x).
    Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """

    return tf.nn.swish(x)


def relu(x):

    return tf.nn.relu(x)


def relu6(x):

    return tf.nn.relu6(x)


def leakyRelu(x):

    return tf.nn.leaky_relu(x)