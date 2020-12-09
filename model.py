import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class SimCLRModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_augmentation = CustomAugment()
        self.global_batch_size = None
        self.embedded_dim = None
    
    @tf.function
    def merge_fn(self, _, per_replica_res):
        return self.distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_res, axis=None)

    def train_step(self, x):
        x1 = self.data_augmentation(x)
        x2 = self.data_augmentation(x)
        x = tf.concat([x1, x2], 0)

        with tf.GradientTape() as tape:
            z = self(x)
            z = tf.math.l2_normalize(z, -1)

            z1, z2 = tf.split(z, 2, 0)
            z = tf.concat([z1, z2], -1)

            replica_context = tf.distribute.get_replica_context()
            replica_id = replica_context.replica_id_in_sync_group
            num_replicas = replica_context.num_replicas_in_sync

            per_replica_res = tf.scatter_nd(
                indices=[[replica_id]],
                updates=[z],
                shape=[num_replicas] + [int(self.global_batch_size / num_replicas), self.embedded_dim * 2])

            z = tf.distribute.get_replica_context().merge_call(self.merge_fn, args=(per_replica_res,))
            z = tf.reshape(z, [-1] + z.shape.as_list()[2:])

            zis, zjs = tf.split(z, 2, -1)
            loss = self.compiled_loss(zis, zjs)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}


class CustomAugment(object):
    def __call__(self, sample):
        # Random flips
        sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)

        # Randomly apply transformation (color distortions) with probability p.
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        sample = self._random_apply(self._color_drop, sample, p=0.2)

        return sample

    def _color_jitter(self, x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x

    def _random_apply(self, func, x, p):
        return tf.cond(
            tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                    tf.cast(p, tf.float32)),
            lambda: func(x),
            lambda: x)
