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
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import models, layers
from keras import optimizers, losses
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard


import config.cfgs as cfgs



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

def saveModel(model, model_path):
    """
    save model
    :param model: model
    :param model_name: model file name
    :return:
    """
    try:
        if os.path.exists(os.path.dirname(cfgs.MODEL_PATH)) is False:
            os.makedirs(os.path.dirname(cfgs.MODEL_PATH))
            print('{0} has been created'.format(model_path))
        # save model
        model.save(model_path)
        print('Successful save model to {0}'.format(model_path))
    except Exception as e:
        raise Exception('Failed save model due to {0}'.format(e))

def seveData(data, data_path):
    """
    save data
    :param obj: object
    :param path: save path
    :return:
    """

    try:
        if os.path.exists(os.path.dirname(cfgs.DATA_PATH)) is False:
            os.makedirs(os.path.dirname(cfgs.DATA_PATH))
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        print('Successful save data to {0}'.format(data_path))
    except Exception as e:
        print('Failed save metric data due to {0}'.format(e))

def vgg16_finetune_net():
    """

    :return:
    """
    target_height, target_width = cfgs.TARGET_SIZE
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(target_height, target_width, 3))
    model = models.Sequential()
    model.add(conv_base)
    # flatten layer
    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=len(cfgs.CLASS_NAMES), activation='sigmoid'))


    # frozen conv_base before block_conv1 ~ bloack_conv4
    model.trainable = True
    trainable_status = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            trainable_status = True
        layer.trainable = trainable_status

    return model


def get_callback():
    """
    get callback list
    :param model:
    :return:
    """
    # check and create log path and model path
    if os.path.exists(cfgs.LOG_PATH) is False:
        os.makedirs(cfgs.LOG_PATH)
        print('Successful create path {0}'.format(cfgs.LOG_PATH))
    if os.path.exists(os.path.dirname(cfgs.MODEL_PATH)) is False:
        os.makedirs(os.path.dirname(cfgs.MODEL_PATH))
        print('Successful create path {0}'.format(os.path.dirname(cfgs.MODEL_PATH)))

    # callback parameter
    # early_stopping = EarlyStopping(monitor='acc',
    #                                min_delta=1.)
    # model_checkpoint = ModelCheckpoint(filepath=cfgs.MODEL_PATH,
    #                                    monitor='val_loss',
    #                                    save_best_only=True)

    tensorboard = TensorBoard(log_dir=cfgs.LOG_PATH,
                              histogram_freq=1)

    callback = [tensorboard]
    return callback



def train(train_data, val_data):

    vgg_model = vgg16_finetune_net()
    vgg_model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                      loss=losses.categorical_crossentropy,
                      metrics=['acc'])
    history = vgg_model.fit_generator(generator=train_data,
                                      steps_per_epoch=cfgs.STEP_PER_EPOCH,
                                      epochs=cfgs.EPOCH,
                                      validation_data=val_data.next(),
                                      validation_steps=cfgs.VAL_STEP,
                                      callbacks=get_callback())

    return vgg_model, history

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


    model, history = train(train_generator, val_generator)

    saveModel(model, cfgs.MODEL_PATH)
    seveData(history.history, cfgs.DATA_PATH)


if __name__ == "__main__":
    main()




