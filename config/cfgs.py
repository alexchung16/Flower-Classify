#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cfgs.py.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/16 下午3:57
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import glob
import pathlib
import numpy as np
import tensorflow as tf
import keras
import time

# tensorflow backend config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))
# -------------------------------system config------------------------------------------
# model path
MODEL_PATH = os.path.join('../output', 'model', 'fine_vgg_{0}.h5'.format(time.time()))
# train data path
DATA_PATH = os.path.join('../output', 'data', 'fine_vgg.pkl')
# log path
LOG_PATH = os.path.join('../output', 'logs')

# data path
# origin dataset
ORIGINAL_DATASET_DIR = '../data'
# separate dataset
BASE_DIR = os.path.join(ORIGINAL_DATASET_DIR, 'flower_split')

# train dataset
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'train')
# validation dataset
VAL_DATA_DIR = os.path.join(BASE_DIR, 'val')

INFERENCE_DATA_PATH = os.path.join('../tools', 'demos')
INFERENCE_OUTPUT_PATH = os.path.join('../tools', 'inference_result')




# -----------------------------train config------------------------------------------
BATCH_SIZE = 32
EPOCH = 60
TARGET_SIZE = (150, 150)
TRAIN_SAMPLE_COUNT = len(list(pathlib.Path(TRAIN_DATA_DIR).glob('*/*.jpg')))
VAL_SAMPLE_COUNT = len(list(pathlib.Path(VAL_DATA_DIR).glob('*/*.jpg')))
STEP_PER_EPOCH = np.ceil(TRAIN_SAMPLE_COUNT / BATCH_SIZE)
VAL_STEP = np.ceil(VAL_SAMPLE_COUNT / BATCH_SIZE)
CLASS_NAMES = [file.name for file in pathlib.Path(TRAIN_DATA_DIR).iterdir() if file.is_dir()]