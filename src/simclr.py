import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from data_augmentation import CustomAugment


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
