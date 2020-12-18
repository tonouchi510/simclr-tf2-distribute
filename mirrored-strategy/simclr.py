import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
from data_augmentation import CustomAugment
import helpers


class TrainSimCLR(object):

    def __init__(self, *args, **kwargs):
        self.distribute_strategy = kwargs["distribute_strategy"]
        self.model = kwargs["model"]
        self.optimizer = kwargs["optimizer"]
        self.criterion = kwargs["criterion"]
        self.emb_dim = kwargs["emb_dim"]
        self.global_batch_size = kwargs["global_batch_size"]
        self.data_augmentation = CustomAugment()
        self.negative_mask = helpers.get_negative_mask(self.global_batch_size)
    
    @tf.function
    def _distribute_da_step(self, dist_x):
        dist_x1 = self.data_augmentation(dist_x)
        dist_x2 = self.data_augmentation(dist_x)
        dist_x = tf.concat([dist_x1, dist_x2], 0)
        return dist_x
    
    @tf.function
    def _distribute_model_step(self, dist_x):
        dist_z = self.model(dist_x)
        dist_z = tf.math.l2_normalize(dist_z)

        dist_z1, dist_z2 = tf.split(dist_z, 2, 0)
        dist_z = tf.concat([dist_z1, dist_z2], -1)
        
        replica_context = tf.distribute.get_replica_context()
        replica_id = replica_context.replica_id_in_sync_group
        num_replicas = replica_context.num_replicas_in_sync

        per_replica_res = tf.scatter_nd(
            indices=[[replica_id]],
            updates=[dist_z],
            shape=[num_replicas] + [int(self.global_batch_size/num_replicas), self.emb_dim*2])

        return per_replica_res
    
    @tf.function
    def _distribute_apply_step(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def train_step(self, dist_batch):
        # Data Augmentation for constractive learning
        dist_da_batch = self.distribute_strategy.run(self._distribute_da_step, args=(dist_batch,))

        with tf.GradientTape() as tape:
            per_replica_res = self.distribute_strategy.run(self._distribute_model_step, args=(dist_da_batch,))
            z = self.distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_res, axis=None)
            z = tf.reshape(z, [-1] + z.shape.as_list()[2:])
            
            zis, zjs = tf.split(z, 2, -1)

            l_pos = sim_func_dim1(zis, zjs)
            l_pos = tf.reshape(l_pos, (self.global_batch_size, 1))
            l_pos /= 0.1 # temperature

            negatives = tf.concat([zjs, zis], axis=0)

            loss = 0
            for positives in [zis, zjs]:
                labels = tf.zeros(self.global_batch_size, dtype=tf.int32)

                l_neg = sim_func_dim2(positives, negatives)
                l_neg = tf.boolean_mask(l_neg, self.negative_mask)
                l_neg = tf.reshape(l_neg, (self.global_batch_size, -1))
                l_neg /= 0.1 # temperature

                logits = tf.concat([l_pos, l_neg], axis=1)
                loss += self.criterion(y_pred=logits, y_true=labels)
            loss = loss / (2 * self.global_batch_size)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.distribute_strategy.run(self._distribute_apply_step, args=(gradients,))

        return loss
