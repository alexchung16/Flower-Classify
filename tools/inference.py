#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/16 下午3:42
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import pathlib
import numpy as np
import config.cfgs as cfgs
from keras.models import load_model
from PIL import Image as pil_image
from keras.preprocessing import image
import matplotlib.pyplot as plt


def predict(model, img_batch=None):
    """"""
    predict_array = model.predict(img_batch)
    # squeeze batch dimension
    predict_array = np.squeeze(predict_array, axis=0)
    predict_label = np.argmax(predict_array)
    predict_prob = predict_array[predict_label]

    return predict_label, predict_prob

def image_preprocess(img_path, target_size, color_mode='rgb'):
    """

    :param img_path:
    :param target_size:
    :param img_type:
    :return:
    """
    img = pil_image.open(img_path)

    # convert channel
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    if color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    # resize
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
        img = img.resize(width_height_tuple,  resample=pil_image.NEAREST)
    # convert tensor to array
    img_array= np.asarray(img, dtype=np.float32)
    # expand dimension
    img_array = np.expand_dims(img_array, axis=0)
    # rescale
    img_array /= 255.

    return img_array


if __name__ == "__main__":
    demo_dir = pathlib.Path(cfgs.INFERENCE_DATA_PATH)

    model = load_model(cfgs.MODEL_PATH)
    model.summary()

    for img_path in demo_dir.iterdir():
        img_batch = image_preprocess(img_path=img_path, target_size=cfgs.TARGET_SIZE)
        # plt.title(img_path.name)
        # plt.imshow(img[0])
        # plt.show()
        predict_label, predict_prob = predict(model=model, img_batch=img_batch)
        print('image_sample:{0},\tpredict result: {1}\t predict probability:{2}'.
              format(img_path, cfgs.CLASS_NAMES[predict_label], predict_prob) )











