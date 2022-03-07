#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: lqhou
@file: air_people.py
@time: 2019/06/11
"""

import numpy
from pandas import read_csv
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

global dataset
global scaler


def get_data(look_back):
    """
    获取训练和测试数据
    """
    global dataset
    global scaler

    # 读取数据
    data_frame = read_csv('international-airline-passengers.csv',
                          usecols=[1], engine='python', skipfooter=3)
    dataset = data_frame.values
    dataset = dataset.astype('float32')

    # 对数据进行归一化处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 划分训练集与测试集,将70%的原始数据作为训练数据,剩下的30%作为测试数据
    train_size = int(len(dataset) * 0.70)
    train_data, test_data = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # 生成训练和测试数据
    train_data_x, train_data_y = create_dataset(train_data, look_back)
    test_data_x, test_data_y = create_dataset(test_data, look_back)

    # 对数据进行Reshape操作，以便输入到RNN模型中,RNN模型的input=[samples,look_back,features],这里features为1.
    train_data_x = numpy.reshape(train_data_x, (train_data_x.shape[0],
                                                look_back, 1))
    test_data_x = numpy.reshape(test_data_x, (test_data_x.shape[0],
                                              look_back, 1))

    return train_data_x, train_data_y, test_data_x, test_data_y


def create_dataset(dataset, look_back):
    """
    构造数据的特征列和类标
    """
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)


def get_model(train_data_x, train_data_y, look_back):
    # 构建一个简单的RNN模型
    rnn_model = tf.keras.Sequential()
    rnn_model.add(tf.keras.layers.SimpleRNN(4, input_shape=(look_back，1)))
    rnn_model.add(tf.keras.layers.Dense(1))

    # 编译、训练模型
    rnn_model.compile(loss='mean_squared_error', optimizer='adam')
    rnn_model.fit(train_data_x, train_data_y, epochs=100, batch_size=5, verbose=1)

    return rnn_model


def show_data(predict_train_data, predict_test_data, look_back):
    global dataset
    global scaler

    # 由于预测的值是标准化后的值，因此需要进行还原
    predict_train_data = scaler.inverse_transform(predict_train_data)
    predict_test_data = scaler.inverse_transform(predict_test_data)

    # 训练数据的预测
    predict_train_data_plot = numpy.empty_like(dataset)
    predict_train_data_plot[:, :] = numpy.nan
    predict_train_data_plot[look_back:len(predict_train_data)
                                      + look_back, :] = predict_train_data

    # 测试数据的预测
    predict_test_data_plot = numpy.empty_like(dataset)
    predict_test_data_plot[:, :] = numpy.nan
    predict_test_data_plot[len(predict_train_data)
                           + 2look_back:len(dataset), :] = predict_test_data

    # 绘制数据
    plt.plot(scaler.inverse_transform(dataset), color='blue', label='Raw data')
    plt.plot(predict_train_data_plot, color='red', label='Train data')
    plt.plot(predict_test_data_plot, color='green', label='Test data')

    # 设置标签
    label = plt.legend(loc='best', ncol=1, fancybox=True)
    label.get_frame().set_alpha(0.5)

    plt.show()


look_back = 1
# 获取预处理过的数据
train_x, train_y, test_x, test_y = get_data(look_back)
# 训练模型
model = get_model(train_x, train_y, look_back)

# 使用训练好的模型进行预测
predict_train_data = model.predict(train_x)
predict_test_data = model.predict(test_x)

# 可视化预测结果
show_data(predict_train_data, predict_test_data, look_back)
