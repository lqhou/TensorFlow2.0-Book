#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: lqhou
@file: text_classification.py
@time: 2019/09/07
"""


import tensorflow as tf
import numpy as np
from data_processing import DataConfig
import datetime
from data_processing import load_data


def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(DataConfig.vocab_size, 16))
    # 使用LSTM的双向循环神经网络
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)))
    # 使用LSTM的单向循环神经网络
    # model.add(tf.keras.layers.LSTM(16))
    # 单向循环神经网络
    # model.add(tf.keras.layers.GRU(16))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    data_path = "./news_data"

    train_x, train_y = load_data(data_path)

    # 随机打乱数据集顺序
    np.random.seed(116)
    np.random.shuffle(train_x)
    np.random.seed(116)
    np.random.shuffle(train_y)

    x_val = train_x[:10000]
    partial_x_train = train_x[10000:]
    y_val = train_y[:10000]
    partial_y_train = train_y[10000:]

    # 设置TensorBoard
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = get_model()
    model.fit(partial_x_train, partial_y_train,
              epochs=40, batch_size=512,
              validation_data=(x_val, y_val),
              verbose=1, callbacks=[tensorboard_callback])
