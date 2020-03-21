#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: lqhou
@file: CIFAR10.py
@time: 2019/08/11
"""


import tensorflow as tf
import numpy as np
import pickle
import os
import datetime


# 设置TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def get_pickled_data(data_path):
    data_x = []
    data_y = []
    with open(data_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        x = data[b'data']
        y = data[b'labels']
        # 将3*32*32的数组变换为32*32*3
        x = x.reshape(10000, 3, 32, 32)\
            .transpose(0, 2, 3, 1).astype('float')
        y = np.array(y)
        data_x.extend(x)
        data_y.extend(y)
    return data_x, data_y


def prepare_data(path):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(5):
        # train_data_path为训练数据的路径
        train_data_path = os.path.join(path, ('data_batch_'+str(i + 1)))
        data_x, data_y = get_pickled_data(train_data_path)
        x_train += data_x
        y_train += data_y
    # 将50000个list型的数据样本转换为ndarray型
    x_train = np.array(x_train)

    # test_data_path为测试文件的路径
    test_data_path = os.path.join(path, 'test_batch')
    x_test, y_test = get_pickled_data(test_data_path)
    x_test = np.array(x_test)

    return x_train, y_train, x_test, y_test


class residual_lock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(residual_lock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=strides,
                                            padding="same")
        # 规范化层：加速收敛，控制过拟合
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        # 规范化层：加速收敛，控制过拟合
        self.bn2 = tf.keras.layers.BatchNormalization()
        # 残差块的第一个卷积层中，卷积核的滑动步长为2时，输出特征图大小减半，
        # 需要对残差块的输入使用步长为2的卷积来进行下采样，从而匹配维度
        if strides != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filters,
                                                       kernel_size=(1, 1),
                                                       strides=strides))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # 匹配维度
        identity = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu = tf.nn.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2)

        output = tf.nn.relu(tf.keras.layers.add([identity, bn2]))

        return output


def build_blocks(filters, blocks, strides=1):
    """组合相同特征图大小的残差块"""
    res_block = tf.keras.Sequential()
    # 添加第一个残差块，每部分的第一个残差块的第一个卷积层，其滑动步长为2
    res_block.add(residual_lock(filters, strides=strides))

    # 添加后续残差块
    for _ in range(1, blocks):
        res_block.add(residual_lock(filters, strides=1))

    return res_block


class ResNet(tf.keras.Model):
    """ResNet模型"""
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.preprocess = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding='same'),
            # 规范化层：加速收敛，控制过拟合
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            # 最大池化：池化操作后，特征图大小减半
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
        ])

        # 组合四个部分的残差块
        self.blocks_1 = build_blocks(filters=64, blocks=3)
        self.blocks_2 = build_blocks(filters=128, blocks=4, strides=2)
        self.blocks_3 = build_blocks(filters=256, blocks=6, strides=2)
        self.blocks_4 = build_blocks(filters=512, blocks=3, strides=2)

        # 平均池化
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # 最后的全连接层，使用softmax作为激活函数
        self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None):
        preprocess = self.preprocess(inputs)
        blocks_1 = self.blocks_1(preprocess)
        blocks2 = self.blocks_2(blocks_1)
        blocks3 = self.blocks_3(blocks2)
        blocks4 = self.blocks_4(blocks3)
        avg_pool = self.avg_pool(blocks4)
        out = self.fc(avg_pool)

        return out


if __name__ == '__main__':
    model = ResNet()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    # 数据集路径
    path = "./cifar-10-batches-py"

    # 数据载入
    x_train, y_train, x_test, y_test = prepare_data(path)
    # 将类标进行one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # 动态设置学习率
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2, patience=5,
        min_lr=0.5e-6)
    callbacks = [lr_reducer, tensorboard_callback]

    # 训练模型
    model.fit(x_train, y_train,
              batch_size=50, epochs=20,
              verbose=1, callbacks=callbacks,
              validation_data=(x_test, y_test),
              shuffle=True)
