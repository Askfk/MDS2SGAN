import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def imfs_loss(pred, gt):
    """

    :param pred: [b, 9, 2704]
    :param gt: [b, 9, 2800]
    :return:
    """

    shape = pred.shape
    n = shape[0] * shape[1] * shape[2]
    gt = gt[:, :, :shape[2]]  # get 2704 points rather than 2800

    mse_loss = K.mean(tf.keras.losses.MSE(gt, pred))
    upper = (2 * n - 1) * K.sum(tf.math.multiply(gt, pred))
    lower = (n + 1) * K.sum(tf.math.multiply(gt, gt))

    loss = mse_loss + upper / lower

    return K.switch(tf.math.is_nan(loss), 0., loss)


def poss_loss(pred, gt):
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = K.mean(loss_func(gt, pred))

    return K.switch(tf.math.is_nan(loss), 0., loss)


def depth_loss(pred, gt):
    loss = K.mean(tf.keras.losses.MSE(gt, pred))

    return K.switch(tf.math.is_nan(loss), 0., loss)


def loc_loss(pred, gt):
    loss = K.mean(tf.keras.losses.MSE(gt, pred))
    return K.switch(tf.math.is_nan(loss), 0., loss)


def feats_loss(pred, gt):
    loss = K.mean(tf.keras.losses.MSE(gt, pred))
    return K.switch(tf.math.is_nan(loss), 0., loss)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Networks_1st import Generator, Discriminator
    from Networks_2nd import Generator as G2
    from DataGenerator import DataGenerator
    import Losses
    from utils import visualize_original_and_decomposed_modals
    from config import Config

    cfg = Config()
    generator = DataGenerator(cfg).generator
    dataset = tf.data.Dataset.from_generator(generator,
                                             (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                             (tf.TensorShape([3, 2800]),
                                              tf.TensorShape([144, 144, 57]),
                                              tf.TensorShape([9, 2800]),
                                              tf.TensorShape([144, 72]),
                                              tf.TensorShape([2]),
                                              tf.TensorShape([None])))
    dataset = dataset.batch(8).shuffle(16)

    g = Generator(cfg)
    d = Discriminator(cfg)
    g2 = G2(cfg)

    for signals, input_tensor, gt_imfs, gt_loc, gt_damage_matrix, gt_depth in dataset.take(2):

        print("--------------------------Seperate Line---------------------------")
        feats, imfs = g(input_tensor)
        pm, dm, lm, pm_logits = d(feats)

        imfs_loss = Losses.imfs_loss(imfs, gt_imfs)
        print("Imfs loss : {}".format(imfs_loss))

        pm_loss = Losses.poss_loss(pm_logits, gt_damage_matrix)
        print("Possibility loss : {}".format(pm_loss))

        dm_loss = Losses.depth_loss(dm, gt_depth)
        print("Depth loss : {}".format(dm_loss))

        loc_loss = Losses.loc_loss(lm, gt_loc)
        print("Localization loss : {}".format(loc_loss))
        visualize_original_and_decomposed_modals(signals, np.exp(gt_imfs * cfg.IMFS_SCALE) - cfg.SHIFT)
        visualize_original_and_decomposed_modals(signals, np.exp(imfs * cfg.IMFS_SCALE) - cfg.SHIFT)

        plt.show()
