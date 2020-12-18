import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.data.ops.readers import TFRecordDatasetV2
from typing import Callable, Any
from typing import List
import json


def get_dataset(dataset_path: str,
                split: str,
                preprocessing: Callable[[Any, int, List], Any],
                global_batch_size: int,
                input_size: int,
                percentage: int = 100) -> TFRecordDatasetV2:

    file_names = tf.io.gfile.glob(f"{dataset_path}/{split}-*.tfrec")

    # Percentage of data used (for finetune)
    if percentage != 100:
        size = int((len(file_names) / 100) * percentage)
        file_names = file_names[:size]

    # Build a pipeline
    option = tf.data.Options()
    option.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(file_names, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    if split == "train":
        dataset = (
            dataset
                .with_options(option)
                .map(lambda x: preprocessing(x, input_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .shuffle(512, reshuffle_each_iteration=True)
                .batch(global_batch_size, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
    else:
        dataset = (
            dataset
                .with_options(option)
                .map(lambda x: preprocessing(x, input_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(global_batch_size, drop_remainder=False)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
    return dataset
