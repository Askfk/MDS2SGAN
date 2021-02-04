"""Generative Adversarial model loss function."""
import tensorflow as tf
import tensorflow.keras.backend as K
from ..config import Config


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multi-class classification,
    always used when data distribution is insufficient.

    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss


def generator_loss_graph(predictions, ground_truth, amplifier=Config.AMPLIFIER, num_modals=Config.NUM_MODALS,
                         loss_func=tf.keras.losses.MSE):
    """As designed, the predictions should be the classification of the
    distractions.

    predictions: [batch, config.SIGNAL_FREQ, config.SIGNAL_PERIOD, config.NUM_STATUS],
                the predicted single mode waves from generator.
    ground_truth_out: [batch, config.SIGNAL_FREQ, config.SIGNAL_PERIOD, config.NUM_STATUS],
                the ground truth single mode waves.
    ground_truth_in: [batch, config.SIGNAL_FREQ, config.SIGNAL_PERIOD, 1],
                the multiple-modes single wave (generator's input), the reduce sum of predictions should be equal to
                the input.
    loss_fuc: MSE or SMISS, etc.

    This loss func is only used in first-stage training.

    To be notable, all the values should be in the normalized coordinates.
    """
    tempre = []
    for i in range(3):
        tempre.append(tf.reduce_sum(predictions[:, :, :, i*num_modals: (i+1)*num_modals], axis=3, keepdims=True))
    predictions = tf.concat(tempre, axis=3)
    loss = K.mean(loss_func(ground_truth / amplifier, predictions))
    final_loss = K.switch(tf.math.is_nan(loss), 0, loss)
    return final_loss


def discriminator_classify_loss_graph(pred_pred_class, gt_class, pred_gt_class=None,
                             loss_func=tf.keras.losses.categorical_crossentropy):
    """As designed, please refer to
    https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb#scrollTo=k6qC-SbjK0yW
    to get concrete detail of this loss.

    pred_pred_class: [batch, config.NUM_CLASSES], the pred classes of the predicted single mode waves from generator.
    pred_gt_class: [batch, config.NUM_CLASSES], the pred classes of the ground truth single mode waves.
    gt_class: [batch, config.NUM_CLASSES], the ground truth classes.

    This loss func will be used in the first-stage training and second-stage training, in first stage, we have both
    pred_pred_class and pred_gt_class while in second stage we have only pred_pred_class.
    """

    if pred_gt_class is not None:
        mode_3_loss = K.mean(loss_func(gt_class, pred_gt_class, from_logits=True))
    else:
        mode_3_loss = 0

    mode_4_loss = K.mean(loss_func(gt_class, pred_pred_class, from_logits=True))

    loss = mode_3_loss + mode_4_loss
    final_loss = K.switch(tf.math.is_nan(loss), 0, loss)

    return final_loss


def discriminator_localize_loss_graph(pred_localization, gt_localization, loss_func=tf.keras.losses.MSE):

    pred_localization = tf.reshape(pred_localization, [-1, Config.LOCAL_HEIGHT, Config.LOCAL_WIDTH])

    loss = K.mean(loss_func(gt_localization, pred_localization))

    return K.switch(tf.math.is_nan(loss), 0, loss)


def transformer_loss_graph(pred, gt,
                           loss_func=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')):

    mask = tf.math.logical_not(tf.math.equal(gt, 0))
    loss_ = loss_func(gt, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


if __name__ == '__main__':
    test_data = tf.random.uniform([6, 96, 96, 12])
    test_gt = tf.random.uniform([6, 96, 96, 3])
    pre = generator_loss_graph(4, test_data, test_gt)
    print(pre)
