# -*-coding:utf-8-*-

import sys
import json
import requests

_IMAGE_SIZE = 64
SERVER_URL = 'http://localhost:8601/v1/models/nsfw:predict'
_LABEL_MAP = {0:'drawings', 1:'hentai', 2:'neutral', 3:'porn', 4:'sexy'}

from PIL import Image
import numpy as np


def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img


def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE))
    img.load()
    data = np.asarray(img, dtype="float32")
    data = standardize(data)
    data = data.astype(np.float16, copy=False)
    return data


def nsfw_predict(image_data):
    pay_load = json.dumps({"inputs": [image_data.tolist()]})
    response = requests.post(SERVER_URL, data=pay_load)
    data = response.json()
    outputs = data['outputs']
    predict_result = {"classes": _LABEL_MAP.get(outputs['classes'][0])}
    predict_result['probabilities'] = {_LABEL_MAP.get(i): l for i, l in enumerate(outputs['probabilities'][0])}
    return predict_result


if __name__ == '__main__':
    image_path = ''
    args = sys.argv
    if len(args) < 2:
        print("usage: python serving_client.py <image_path>")
    image_data = load_image(image_path)
    predict = nsfw_predict(image_data)
    print(predict)
