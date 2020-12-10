# simclr-tf2-distribute
Implementation of [SimCLR](https://arxiv.org/abs/2002.05709) with TensorFlow 2 / `tf.distribute.Strategy`.

<div align="center">
  <img src="https://github.com/tonouchi510/simclr-tf2-distribute/blob/main/figs/illustration-of-the-proposed-SimCLR-framework.gif" width="550px">
</div>

<br>

SimCLR needs to be trained in a huge batch size, so it is practically necessary to support distributed learning.

<div align="center">
  <img src="https://github.com/tonouchi510/simclr-tf2-distribute/blob/main/figs/simclr-figure-9.png" width="500px">
</div>

This time, use [TPU Strategy](https://www.tensorflow.org/guide/distributed_training#tpustrategy)(If you use 8-cores, CloudTPU is more inexpensive than GPU).
If the GPU is better, you can use [MirroredStrategy](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy).

If you want to know about `tf.distribute.Strategy`, see here https://www.tensorflow.org/guide/distributed_training


## Acknowledgements
I reuse some of the code from [sayakpaul/SimCLR-in-TensorFlow-2](https://github.com/sayakpaul/SimCLR-in-TensorFlow-2) in this implementation.  
Thanks for showing me the reference implementation by tf2.

The differences in my implementation are the following points,
- Supports distributed training, to train with huge batch sizes.
- Instead of a custom training loop, use `tf.keras.Model.fit`.
  - Take advantage of the tf.keras.Model class.
  - Easy to use various callbacks, saving the state of the optimizer during training, etc.
- Use cloud-tpu for training. 

## Usage
#### Pre-required
To train with CloudTPU, you need to prepare for the following,
- Setup GCP project and enable CloudTPU
- Convert the dataset to TFRecord files and upload to GCS.
  - When use TPU, all the files used during training need to be put in GCS.

#### Pretrain

```
$ python pretrain.py --global_batch_size=1024 --epochs=50 --learning_rate=0.0001 \
    --temperature=0.1 --embedded_dim=128 --dataset=gs://{{bucket-name}}/{{tfrecord_dir}} \
    --model="resnet" --job_dir=gs://{{bucket-name}}/{{job_dir}}
```

#### Finetune

```
$ python finetune.py --global_batch_size=1024 --epochs=50 --learning_rate=0.0001 --proj_head=1 \
    --percentage=10 --num_classes=1000 --dataset=gs://{{bucket-name}}/{{tfrecord_dir}} \
    --job_dir=gs://{{bucket-name}}/{{job_dir}}
```