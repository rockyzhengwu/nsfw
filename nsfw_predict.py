#-*-coding:utf-8-*-

import os
import sys

import numpy as np
from PIL import Image
import tensorflow as tf


_MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/models/1547856517')

_IMAGE_SIZE = 64
_BATCH_SIZE = 128

_LABEL_MAP = {0:'drawings', 1:'hentai', 2:'neutral', 3:'porn', 4:'sexy'}

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img

def load_image( infilename ) :
    img = Image.open( infilename )
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE))
    img.load()
    data = np.asarray( img, dtype=np.float32 )
    data = standardize(data)
    return data

def predict(image_path):
    with tf.Session() as sess:
        graph = tf.get_default_graph();
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], _MODEL_DIR)
        inputs = graph.get_tensor_by_name("input_tensor:0")
        probabilities_op = graph.get_tensor_by_name('softmax_tensor:0')
        class_index_op = graph.get_tensor_by_name('ArgMax:0')

        image_data = load_image(image_path)
        probabilities, class_index = sess.run([probabilities_op, class_index_op],
                                              feed_dict={inputs: [image_data] * _BATCH_SIZE})

        probabilities_dict = {_LABEL_MAP.get(i): l for i, l in enumerate(probabilities[0])}
        pre_label = _LABEL_MAP.get(class_index[0])
        result = {"class": pre_label, "probability": probabilities_dict}
        return result


if __name__ == '__main__':
    argv = sys.argv
    if(len(argv) < 2):
        print("usage: python nsfw_predict <image_path>")
    image_path = argv[1]
    print()
    res = predict(image_path)
    print(res)


