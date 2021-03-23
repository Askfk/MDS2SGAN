import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def imfs_loss(pred, gt):
    n = pred.shape[1]

    mse_loss = K.mean(tf.keras.losses.MSE(pred, gt))
    upper = (2 * n - 1) * K.sum(tf.math.multiply(pred, gt))
    lower = (n + 1) * K.sum(tf.math.multiply(gt, gt))

    loss = mse_loss + upper / lower

    return K.switch(tf.math.is_nan(loss), 0, loss)


def poss_loss(pred, gt):
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = K.mean(loss_func(pred, gt))

    return K.switch(tf.math.is_nan(loss), 0, loss)


def depth_loss(pred, gt):
    loss = K.mean(tf.keras.losses.MSE(pred, gt))

    return K.switch(tf.math.is_nan(loss), 0, loss)


def loc_loss(pred, gt):
    loss = K.mean(tf.keras.losses.MSE(pred, gt))
    return K.switch(tf.math.is_nan(loss), 0, loss)
