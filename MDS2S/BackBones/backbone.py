"""Build efficient bottom-up networks."""

from . import EfficientNet, ResNet, VGGNet


def build_backbone_net_graph(architecture, config=None):
    """
    Build basic feature extraction networks.
    :param config:
    :param input_tensor: Input of the basic networks, should be a tensor or tf.keras.layers.Input
    :param architecture: The architecture name of the basic network.
    # :param weights: Whether download and initialize weights from the pre-trained weights,
    #                 could be either 'imagenet', (pre-training on ImageNet)
    #                                 'noisy-student',
    #                                 'None' (random initialization)，
    #                                 or the path to the weights file to be loaded。
    :return: Efficient Model and corresponding endpoints.
    """

    if architecture.startswith('efficientnet'):
        model = EfficientNet.EfficientNetX(config, architecture)
        return model
    elif architecture.startswith('resnet'):
        model = ResNet.ResNetX(config, architecture)
        # model = model.build_model(input_tensor)
        return model
    elif architecture.startswith('vgg'):
        model = VGGNet.VGGNetX(config, architecture)
        return model
    else:
        return [None] * 5


if __name__ == '__main__':
    from config import Config
    import tensorflow as tf
    import numpy as np
    config = Config()
    model = build_backbone_net_graph('efficientnet-b3', config)
    input_tensor = np.random.random([1, 96, 96, 3])
    input_tensor = tf.convert_to_tensor(input_tensor, tf.float32)
    b = model(input_tensor, False)
    for o in b:
        print(o.shape)
    [o1, o2, _, o4, _] = b
    o1 = tf.reshape(o1, [o1.shape[0], o1.shape[1] ** 2, o1.shape[3]])
    o2 = tf.reshape(o2, [o2.shape[0], o2.shape[1] ** 2, o2.shape[3]])
    o4 = tf.reshape(o4, [o4.shape[0], o4.shape[1] ** 2, o4.shape[3]])
    o = tf.concat([o1, o2, o4], axis=1)
    o = tf.reshape(o, [o.shape[0], int(o.shape[1]**0.5), int(o.shape[1]**0.5), o.shape[2]])
    print(o.shape)

