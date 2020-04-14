#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : VGG16.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/14 上午10:55
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import glob
import tensorflow as tf

if __name__ == "__main__":
    import os
    import glob
    import pathlib
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from PIL import Image
    import matplotlib.pyplot as plt

    data_dir = pathlib.Path('./data')

    # get flower category
    class_names = np.array([item.name for item in data_dir.glob('*/*') if item.name != "LICENSE.txt"])
    # count image number
    image_count = len(list(data_dir.glob('*/*/*.jpg')))
    print("flower category: {0}".format(class_names))
    print("number image: {0}".format(image_count))

    # dataset is divided into three sets: 1)train set 2)validation set 3)test set
    # define train_dir
    train_dir = os.path.join(data_dir, 'train')

    # define validation dir
    val_dir = os.path.join(data_dir, 'val')

    # define test dir
    test_dir = os.path.join(data_dir, 'test')

    dataset = {}
    for category in class_names:
        dataset[category] = data_dir.glob('*/{0}/*.jpg'.format(category))

    # get category count
    # class_count = [len(list(data)) for category, data in dataset.items()]
    # print(class_count)

    print(dataset['sunflowers'])

    train_data = []
    for index, category in enumerate(class_names):
        for img_path in list(dataset[category]):
            # image path and class label
            train_data.append([img_path, index])
    # get DataFrame from List
    train_data = pd.DataFrame(train_data, columns=['images', 'labels'], index=None)

    # shuffle the dataset
    train_data = train_data.sample(frac=1.).reset_index(drop=True)
    print(train_data.head())
