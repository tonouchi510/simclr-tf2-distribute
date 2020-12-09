import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

from general import get_train_ds
from model import SimCLRModel
from data import read_tfrecord, CustomAugment
from losses import simclr_loss_func
from general import Training

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
    'learning_rate', 0.0001,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature coefficient for simclr training.')

flags.DEFINE_integer(
    'embedded_dim', 128,
    'n_dim for projection_head_3.')

flags.DEFINE_float(
    'proj_head', 1,
    'Projection_Head using finetune.')

flags.DEFINE_string(
    'dataset', 'gs://{{bucket-name}}/{{tfrecord_dir}}',
    'Directory where dataset is stored.')

flags.DEFINE_string(
    'model', 'resnet',
    'Model type for training simclr.')

flags.DEFINE_string(
    'job_dir', 'gs://{{bucket-name}}/{{job_dir}}',
    'GCS path for training job.')


def build_model(model_type: str = "", n_dim: int = 128) -> (model: any, input_size: int):
    """Building model and return params.

    Args:
        model_type (str): Initially resnet or efficientnet is available.
        n_dim (int): Number of embedded dimensions of the projection_head_3.

    Returns:
        model (any): Object of SimCLRModel class.
        input_size (int): Image size for input to model.
    """
    if model_type == 'resnet':
        input_size = 112
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(112, 112, 3))
    elif model_type == 'efficientnet':
        input_size = 224
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))
    else:
        print("Invariant model_type.")
        return

    base_model.trainable = True
    inputs = Input((input_size, input_size, 3))
    h = base_model(inputs, training=True)
    h = GlobalAveragePooling2D()(h)

    projection_1 = Dense(n_dim * 4)(h)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(n_dim * 2)(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(n_dim)(projection_2)

    model = SimCLRModel(inputs, projection_3)
    # set property
    model.global_batch_size = FLAGS.global_batch_size
    model.embedded_dim = FLAGS.embedded_dim
    return model, input_size


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

    return image


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Setup tpu-cluster
    cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster)
    tf.tpu.experimental.initialize_tpu_system(cluster)
    distribute_strategy = tf.distribute.TPUStrategy(cluster)

    with distribute_strategy.scope():
        simclr_model, input_size = build_model(model_type=FLAGS.model, n_dim=FLAGS.embedded_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    simclr_model.compile(loss=simclr_loss_func,
                         optimizer=optimizer,
                         metrics=None)
    simclr_model.summary()

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{FLAGS.job_dir}/logs", histogram_freq=1)
    callbacks = [tboard_callback]

    train_ds = get_dataset(FLAGS.dataset, "train", read_tfrecord, FLAGS.global_batch_size, input_size)
    for epoch in range(FLAGS.epochs):
        simclr_model.fit(train_ds, callbacks=callbacks, initial_epoch=epoch, epochs=epoch+1)
        simclr_model.save(f"{FLAGS.job_dir}/checkpoints/{epoch+1}", include_optimizer=True)
    
    simclr_model.save(f"{FLAGS.job_dir}/saved_model", include_optimizer=False)


if __name__ == '__main__':
    app.run(main)
