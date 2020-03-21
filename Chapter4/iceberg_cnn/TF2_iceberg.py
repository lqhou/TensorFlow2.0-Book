#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: lqhou
@file: TF2_iceberg.py
@time: 2019/03/10
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


def data_preprocess(path, more_data):
    # 读取数据
    data_frame = pd.read_json(path)

    # 获取图像数据
    images = []
    for _, row in data_frame.iterrows():
        # 将一维数据转为75x75的二维数据
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        images.append(np.dstack((band_1, band_2, band_3)))
    if more_data:
        # 扩充数据集
        images = create_more_data(np.array(images))

    # 获取类标
    labels = np.array(data_frame['is_iceberg'])
    if more_data:
        # 扩充数据集后，类标也需要相应扩充
        labels = np.concatenate((labels, labels, labels, labels, labels, labels))

    return np.array(images), labels


def create_more_data(images):
    # 扩充数据
    image_rot90 = []
    image_rot180 = []
    image_rot270 = []
    img_lr = []
    img_ud = []

    for i in range(0, images.shape[0]):
        band_1 = images[i, :, :, 0]
        band_2 = images[i, :, :, 1]
        band_3 = images[i, :, :, 2]

        # 旋转90度
        band_1_rot90 = np.rot90(band_1)
        band_2_rot90 = np.rot90(band_2)
        band_3_rot90 = np.rot90(band_3)
        image_rot90.append(np.dstack((band_1_rot90, band_2_rot90, band_3_rot90)))

        # 旋转180度
        band_1_rot180 = np.rot90(band_1_rot90)
        band_2_rot180 = np.rot90(band_2_rot90)
        band_3_rot180 = np.rot90(band_3_rot90)
        image_rot180.append(np.dstack((band_1_rot180, band_2_rot180, band_3_rot180)))

        # 旋转270度
        band_1_rot270 = np.rot90(band_1_rot180)
        band_2_rot270 = np.rot90(band_2_rot180)
        band_3_rot270 = np.rot90(band_3_rot180)
        image_rot270.append(np.dstack((band_1_rot270,
                                       band_2_rot270, band_3_rot270)))

        # 左右翻转
        lr1 = np.flip(band_1, 0)
        lr2 = np.flip(band_2, 0)
        lr3 = np.flip(band_3, 0)
        img_lr.append(np.dstack((lr1, lr2, lr3)))

        # 上下翻转
        ud1 = np.flip(band_1, 1)
        ud2 = np.flip(band_2, 1)
        ud3 = np.flip(band_3, 1)
        img_ud.append(np.dstack((ud1, ud2, ud3)))

    rot90 = np.array(image_rot90)
    rot180 = np.array(image_rot180)
    rot270 = np.array(image_rot270)
    lr = np.array(img_lr)
    ud = np.array(img_ud)
    images = np.concatenate((images, rot90, rot180, rot270, lr, ud))

    return images


# 定义模型
def get_model():
    # 建立一个序贯模型
    model = tf.keras.Sequential()

    # 第一个卷积层
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu',
                            input_shape=(75, 75, 3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Dropout(0.2))

    # 第二个卷积层
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Dropout(0.2))

    # 第三个卷积层
    model.add(layers.Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Dropout(0.2))

    # 第四个卷积层
    model.add(layers.Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Dropout(0.2))

    # 将上一层的输出特征映射转化为一维数据，
    # 以便进行全连接操作
    model.add(layers.Flatten())

    # 第一个全连接层
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # 第二个全连接层
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # 第三个全连接层
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.0001),
                  metrics=['accuracy'])
    # 打印出模型的概况信息
    model.summary()

    return model


# 数据预处理
train_x, train_y = data_preprocess('./data/train.json', more_data=True)

# 初始化模型
cnn_model = get_model()

# 模型训练
cnn_model.fit(train_x, train_y, batch_size=25,
              epochs=100, verbose=1, validation_split=0.2)
