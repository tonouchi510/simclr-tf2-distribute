import tensorflow as tf
from absl import flags
import helpers

cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

FLAGS = flags.FLAGS


def simclr_loss_func(zis, zjs):
    negative_mask = helpers.get_negative_mask(FLAGS.global_batch_size)
    l_pos = _dot_simililarity_dim1(zis, zjs)
    l_pos = tf.reshape(l_pos, (FLAGS.global_batch_size, 1))
    l_pos /= FLAGS.temperature

    negatives = tf.concat([zjs, zis], axis=0)

    loss = 0
    for positives in [zis, zjs]:
        labels = tf.zeros(FLAGS.global_batch_size, dtype=tf.int32)

        l_neg = _dot_simililarity_dim2(positives, negatives)
        l_neg = tf.boolean_mask(l_neg, negative_mask)
        l_neg = tf.reshape(l_neg, (FLAGS.global_batch_size, -1))
        l_neg /= FLAGS.temperature

        logits = tf.concat([l_pos, l_neg], axis=1)
        loss += criterion(y_pred=logits, y_true=labels)
    loss = loss / (2 * FLAGS.global_batch_size)
    return loss


def _cosine_simililarity_dim1(x, y):
    v = cosine_sim_1d(x, y)
    return v


def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return v


def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v
