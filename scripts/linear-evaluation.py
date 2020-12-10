import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'batch_size', 512,
    'Batch size for training/eval.')

flags.DEFINE_string(
    'job_dir', None,
    'GCS path for training job management.')

flags.DEFINE_integer(
    'embedded_dim', 512,
    'Number of dimensions of extracted feature.')

flags.DEFINE_string(
    'num_classes', None,
    'Num of class label.')


def build_model(*, num_classes: int = 0, n_dim = 512) -> any:

    inputs = tf.keras.Input(shape=(n_dim,))
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(inputs)
    model = tf.keras.Model(inputs, outputs)

    return model


def read_tfrecord(example, _, label_list):
    features = {
        "feature":  tf.io.FixedLenFeature([], tf.string),
        "label":    tf.io.FixedLenFeature([], tf.string),
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    z = tf.io.parse_tensor(example["feature"], tf.float32)

    label = example['label']
    label_num = tf.where(label_list==label)[0]
    return z, label_num


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    model = build_model(num_classes=FLAGS.num_classes, n_dim=FLAGS.embedded_dim)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    model.summary()

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{FLAGS.job_dir}/linear-evaluation/logs", histogram_freq=1)
    callbacks = [tboard_callback]

    dataset_path = f"{FLAGS.job_dir}/extract_feature"

    train_ds = get_dataset(dataset_path, "train", read_tfrecord, FLAGS.global_batch_size, input_size, FLAGS.percentage)
    valid_ds = get_dataset(dataset_path, "valid", read_tfrecord, FLAGS.global_batch_size, input_size, FLAGS.percentage)

    model.fit(train_ds, validation_data=valid_ds, callbacks=callbacks, epochs=FLAGS.epochs)
    model.save(f"{FLAGS.job_dir}/linear-evaluation/saved_model", include_optimizer=False)


if __name__ == '__main__':
    app.run(main)
