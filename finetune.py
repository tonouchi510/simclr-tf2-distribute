from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

from data import get_dataset

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'global_batch_size', 2048,
    'Batch size for training/eval before distribution.')

flags.DEFINE_integer(
    'epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_float(
    'learning_rate', 0.001,
    'Initial learning rate')

flags.DEFINE_float(
    'proj_head', 1,
    'Projection_Head using finetune.')

flags.DEFINE_string(
    'dataset', None,
    'Directory where dataset is stored.')

flags.DEFINE_string(
    'job_dir', None,
    'GCS path for training job management.')

flags.DEFINE_string(
    'pretrain_model', None,
    'pretrain model path.')

flags.DEFINE_integer(
    'percentage', 100,
    'percentage of dataset for using finetune.')

flags.DEFINE_string(
    'num_classes', None,
    'GCS path for saving artifacts.')


def build_model(num_classes: int) -> (any, int):
    base_model = tf.keras.models.load_model(FLAGS.pretrain_model)
    base_model.trainable = True

    feature = None
    if FLAGS.proj_head == 0:
        feature = base_model.layers[-6].output
    elif FLAGS.proj_head == 1:
        feature = base_model.layers[-4].output
    elif FLAGS.proj_head == 2:
        feature = base_model.layers[-2].output

    head = Dense(num_classes, name="dense_head", activation="softmax")(feature)
    model = Model(base_model.input, head)
    return model, model.input_shape[1]


def read_tfrecord(example, size: int):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    resize_crit = (w * size) / (h * size)
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(image, [w*size/w, h*size/w]),  # if true
                    lambda: tf.image.resize(image, [w*size/h, h*size/h])   # if false
                    )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - size) // 2, (nh - size) // 2, size, size)

    return image, example["label"]


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Setup tpu-cluster
    cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster)
    tf.tpu.experimental.initialize_tpu_system(cluster)
    distribute_strategy = tf.distribute.TPUStrategy(cluster)

    with distribute_strategy.scope():
        model, input_size = build_model(num_classes=FLAGS.num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    model.summary()

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{FLAGS.job_dir}/logs", histogram_freq=1)
    callbacks = [tboard_callback]

    train_ds = get_dataset(FLAGS.dataset, "train", read_tfrecord, FLAGS.global_batch_size, input_size, FLAGS.percentage)
    valid_ds = get_dataset(FLAGS.dataset, "valid", read_tfrecord, FLAGS.global_batch_size, input_size, FLAGS.percentage)
    for epoch in range(FLAGS.epochs):
        model.fit(train_ds, validation_data=valid_ds, callbacks=callbacks, initial_epoch=epoch, epochs=epoch+1)
        model.save(f"{FLAGS.job_dir}/checkpoints/{epoch+1}", include_optimizer=True)
    
    model.save(f"{FLAGS.job_dir}/saved_model", include_optimizer=False)


if __name__ == '__main__':
    app.run(main)
