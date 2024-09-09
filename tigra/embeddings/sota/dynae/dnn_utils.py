import tensorflow as tf
from keras import losses
from tensorflow.python.keras.utils.losses_utils import ReductionV2


class WeightedMSEX(losses.Loss):
    def __init__(self, num_nodes, reduction=ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self._num_nodes = num_nodes

    def call(self, y_true, y_pred):
        """Hack: This fn doesn't accept additional arguments.
                We use y_true to pass them.
        y_pred: Contains x_hat - x_pred
        y_true: Contains b
        """
        return tf.reduce_sum(
            tf.square(y_pred * y_true[:, 0 : self._num_nodes]), axis=-1
        )
