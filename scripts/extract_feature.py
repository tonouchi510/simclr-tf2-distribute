import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.python.framework.ops import Tensor
import numpy as np
from absl import app
from absl import flags
from google.cloud import storage
import os

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'batch_size', 512,
    'batch size for predict.')

flags.DEFINE_float(
    'proj_head', 1,
    'Projection_Head using finetune.')

flags.DEFINE_string(
    'dataset', None,
    'Directory where dataset is stored.')

flags.DEFINE_string(
    'job_dir', None,
    'GCS path for job management.')

flags.DEFINE_string(
    'le_target', None,
    'Target model for linear-evaluation (pretrain or finetune).')


def get_dataset(tfrecord_filepath: str, input_size: int):
    file_names = tf.io.gfile.glob(tfrecord_filepath)

    option = tf.data.Options()
    option.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset(file_names, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = (
        dataset
            .with_options(option)
            .map(lambda x: read_tfrecord(x, input_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(FLAGS.batch_size, drop_remainder=False)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


def read_tfrecord(example: Tensor, size: int):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.int64),  # one bytestring
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    # Resize image
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

    label = example['label']
    return image, label


def to_tfrecord(feature_bytes, label):
    def _bytestring_feature(list_of_bytestrings):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

    def _int_feature(list_of_ints):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

    feature = {
        "feature":  _bytestring_feature([feature_bytes]),
        "label":    _int_feature([label]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(argv):
    """
    Extract image features by trained model.
    Download the tfrecord of the image data set, convert it to the feature tfrecord, and upload it.
    """
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    base_model = tf.keras.models.load_model(f"{FLAGS.job_dir}/{FLAGS.le_target}/saved_model")
    if FLAGS.le_target == "pretrain":
        if FLAGS.proj_head == 0:
            model = Model(base_model.inputs, base_model.layers[-6].output)
        elif FLAGS.proj_head == 1:
            model = Model(base_model.inputs, base_model.layers[-4].output)
        elif FLAGS.proj_head == 2:
            model = Model(base_model.inputs, base_model.layers[-2].output)
        else:
            model = Model(base_model.inputs, base_model.output)
    elif FLAGS.le_target == "finetune":
        model = Model(base_model.inputs, base_model.layers[-2].output)
    else:
        exit(-1)
    model.trainable = False

    input_size = model.input_shape[1]
    model.summary()

    bucket_name = FLAGS.dataset.split("//")[-1].split("/")[0]
    dataset_dir = FLAGS.dataset.split("/")[-1]

    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=f"datasets/{dataset_dir}")
    filelist = []
    for b in blobs:
        filelist.append(os.path.basename(b.name))

    for filename in filelist:
        # Process TFRecord files one by one.
        dataset = get_dataset(f"{FLAGS.dataset}/{filename}", input_size)

        features, labels = [], []
        for image, label in dataset:
            labels.extend(label.numpy())
            features.extend(model.predict(image))

        filepath = f"{FLAGS.job_dir}/extract_feature/{filename}"
        with tf.io.TFRecordWriter(filepath) as out_file:
            for i in range(len(features)):
                example = to_tfrecord(tf.io.serialize_tensor(features[i]).numpy(), labels[i])
                out_file.write(example.SerializeToString())
        print(f"Wrote {filepath}")


if __name__ == '__main__':
    app.run(main)
