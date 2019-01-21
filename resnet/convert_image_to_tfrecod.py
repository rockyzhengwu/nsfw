#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import math
import sys
import tensorflow as tf
import random
import imghdr

_RANDOM_SEED = 0
_TRAIN_NUM_SHARDS = 200

class ImageReader(object):

  def __init__(self):
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
    self._decode_png = tf.image.decode_png(self._decode_jpeg_data, channels = 3)
    self._decode_gif = tf.image.decode_gif(self._decode_jpeg_data)

  def read_image_dims(self, sess, image_data, image_type):
    if image_type == "jpeg" or image_type == "jpg":
      image = self.decode_jpeg(sess, image_data)
    else:
      image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def is_jpeg(self, sess, image_data):
    r = tf.image.is_jpeg(image_data)
    b = sess.run(r)
    return b

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png, feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'nsfw_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation', 'test']
  if split_name == 'train':
    num_shards = _TRAIN_NUM_SHARDS
  else:
    num_shards = 10

  num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))
  print(num_per_shard)
  total_skip = 0
  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            image_type = imghdr.what(filenames[i])
            image_type = image_type if image_type else 'jpg'
            try:
              height, width = image_reader.read_image_dims(sess, image_data, image_type)
            except Exception as e:
              print(filenames[i])
              print(e)
              continue

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()
  print(total_skip)

def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _get_filenames_and_classes(dataset_dir, mode):
  flower_root = os.path.join(dataset_dir, mode)
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)

_NUM_VALIDATION = 4000


def main(dataset_dir):
  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir, 'train')
  test_photo_filenames, test_class_names = _get_filenames_and_classes(dataset_dir, 'test')
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  _convert_dataset('train', photo_filenames, class_names_to_ids, dataset_dir)
  _convert_dataset('validation', test_photo_filenames, class_names_to_ids, dataset_dir)


if __name__ == "__main__":
  """
  """
  dataset_dir = sys.argv[1]
  main(dataset_dir)

