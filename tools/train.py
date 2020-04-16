#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/16 下午3:42
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import models, layers
from keras import optimizers, losses
from keras_preprocessing.image import ImageDataGenerator

import config.cfgs as cfgs

# tensorflow backend config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))




# image generation
def image_generate(train_dir, val_dir, batch_size=16, target_size=(150, 150), classes_name=None, class_model='categorical'):
    """
    image data augmentation
    Note： Only augmentation train dataset but validation dataset
    :return:
    """
    train_data_generate = ImageDataGenerator(rescale=1./255,
                                             rotation_range=40,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             fill_mode='nearest')

    # do not augmentation validation data
    val_data_generate = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generate.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_model,
        classes=classes_name,
    )

    val_generator = val_data_generate.flow_from_directory(
        directory=val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_model,
        classes = classes_name,
    )

    return train_generator, val_generator


def show_batch(image_batch, label_batch, class_names):
    """
    show batch to check the accuracy of generate data
    :param image_batch:
    :param label_batch:
    :return:
    """
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(np.array(class_names)[label_batch[n] == 1][0].title())
        plt.axis('off')
    plt.show()


def main():
    train_generator, val_generator = image_generate(train_dir=cfgs.TRAIN_DATA_DIR,
                                                    val_dir=cfgs.VAL_DATA_DIR,
                                                    batch_size=cfgs.BATCH_SIZE,
                                                    target_size=cfgs.TARGET_SIZE,
                                                    classes_name=cfgs.CLASS_NAMES)
    class_names = cfgs.CLASS_NAMES
    class_indices = train_generator.class_indices
    print(class_names, class_indices)

    # get data
    train_img, train_label = next(train_generator)
    val_img, val_label = next(val_generator)

    show_batch(train_img, train_label, class_names)
    show_batch(val_img, val_label, class_names)

    # show data
    # plt.imshow(train_img[0])
    # plt.imshow(val_img[0])
    # plt.show()








if __name__ == "__main__":
    main()





