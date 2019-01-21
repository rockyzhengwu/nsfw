#-*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from resnet import resnet_model
from resnet import resnet_run_loop

_NUM_CHANNELS = 3
_NUM_CLASSES = 5

# The record is the image plus a one-byte label

_NUM_IMAGES = {
    'train': 230944,
    'validation': 19448,
}

DATASET_NAME = 'nsfw'

_IMAGE_SIZE = 224
_HEIGHT = 224
_WIDTH = 224
###############################################################################
# Data processing
###############################################################################

def get_filenames(is_training, data_dir):
    file_names = []
    if is_training:
        pattern = 'nsfw_train_.*.tfrecord'
    else:
        pattern = 'nsfw_validation_.*.tfrecord'
    for top, dis, files in os.walk(data_dir):
        for name in files:
            if re.match(pattern, name):
                file_names.append(os.path.join(top, name))
    return file_names

def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _IMAGE_SIZE + 8, _IMAGE_SIZE + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def parse_record(raw_record, is_training, dtype ):
    print(dtype)
    image_feature_description = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/encoded': tf.FixedLenFeature([], tf.string),
    }

    parsed = tf.parse_single_example(raw_record, image_feature_description)
    image = parsed['image/encoded']
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype)
    image.set_shape([None, None, 3])
    image = tf.image.resize_images(image, [_IMAGE_SIZE, _IMAGE_SIZE])
    image  = preprocess_image(image, is_training)
    label = parsed['image/class/label']
    label = tf.cast(label, tf.int32)
    image = tf.cast(image, dtype)
    return image, label


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None, dtype=tf.float32):
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.repeat(num_epochs)

    if is_training and num_gpus and examples_per_epoch:
        total_examples = num_epochs * examples_per_epoch
        # Force the number of batches to be divisible by the number of devices.
        # This prevents some devices from receiving batches while others do not,
        # which can lead to a lockup. This case will soon be handled directly by
        # distribution strategies, at which point this .take() operation will no
        # longer be needed.
        total_batches = total_examples // batch_size // num_gpus * num_gpus
        dataset.take(total_batches * batch_size)

    # Parse the raw records into images and labels. Testing has shown that setting
    # num_parallel_batches > 1 produces no improvement in throughput, since
    # batch_size is almost always much greater than the number of CPU cores.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=False))

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset



def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None, dtype=tf.float32):
    filenames = get_filenames(is_training, data_dir)
    print(filenames)

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=500,
        parse_record_fn=parse_record,
        num_epochs=num_epochs,
        num_gpus=num_gpus,
        examples_per_epoch=_NUM_IMAGES['train'] if is_training else None,
        dtype=dtype
    )
    return dataset

def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)



###############################################################################
# Running the model
###############################################################################
class Model(resnet_model.Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self,
               resnet_size,
               data_format=None,
               num_classes=_NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(Model, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def nsfw_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  features = tf.reshape(features, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS])

  if params['fine_tune']:
    warmup = False
    base_lr = .1
  else:
    warmup = True
    base_lr = .128

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'],
      batch_denom=256,
      num_images=_NUM_IMAGES['train'],
      boundary_epochs=[30, 60, 80, 90],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
      warmup=warmup,
      base_lr=base_lr)

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(_):
    return True

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=Model,
      resnet_size=params['resnet_size'],
      weight_decay=weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=loss_filter_fn,
      dtype=params['dtype'],
      fine_tune=params['fine_tune']
  )


def set_defaults(**kwargs):
  for key, value in kwargs.items():
    flags.FLAGS.set_default(name=key, value=value)


def define_nsfw_flags():
  resnet_run_loop.define_resnet_flags()
  flags.adopt_module_key_flags(resnet_run_loop)

  set_defaults(
      data_dir='',
      model_dir='',
      resnet_size='50',
      train_epochs=50,
      epochs_between_evals=50,
      batch_size=128)


def run_nsfw(flags_obj):
  input_function = input_fn
  resnet_run_loop.resnet_main(
      flags_obj,
      nsfw_model_fn,
      input_function,
      DATASET_NAME,
      shape=[_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS])


def main(_):
  run_nsfw(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_nsfw_flags()
  absl_app.run(main)
