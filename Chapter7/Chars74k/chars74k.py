import tensorflow as tf
from tensorflow.keras import layers
import datetime
import numpy as np
from PIL import Image
import os


def get_dataset(path):
    """获取数据集"""
    data_x = []
    data_y = []

    # 获取当前路径下所有文件夹（或文件）
    folder_name = os.listdir(path)

    # 循环遍历每个文件夹
    for i in folder_name:
        file_path = os.path.join(path, i)

        # 取文件夹名后三位整数作为类标
        label = int(i[-3:])

        # 获取当前文件夹下的所有图片文件
        filenames = os.listdir(file_path)

        for filename in filenames:
            # 组合得到每张图片的路径
            image_path = os.path.join(file_path, filename)

            # 读取图片
            image = Image.open(image_path)
            # 将image对象转为numpy数组
            width, height = image.size
            image_matrix = np.reshape(image, [width * height * 3])

            img1 = image.rotate(30)
            img2 = image.rotate(60)
            img3 = image.rotate(90)

            data_x.append(image_matrix)
            data_y.append(label)

            image_matrix = np.reshape(img1, [width * height * 3])
            data_x.append(image_matrix)
            data_y.append(label)
            image_matrix = np.reshape(img2, [width * height * 3])
            data_x.append(image_matrix)
            data_y.append(label)
            image_matrix = np.reshape(img3, [width * height * 3])
            data_x.append(image_matrix)
            data_y.append(label)

    return data_x, data_y


def vgg13_model(input_shape, classes):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, 3, 1, input_shape=input_shape,
                            padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.Conv2D(64, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.Conv2D(128, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.Conv2D(256, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',
                            activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation='softmax'))

    # 模型编译
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    path = './chars74k_data'
    data_x, data_y = get_dataset(path)

    train_x = np.array(data_x).reshape(-1, 224, 224, 3)
    train_y = [i - 1 for i in data_y]

    train_y = tf.keras.utils.to_categorical(train_y, 62)

    # 随机打乱数据集顺序
    np.random.seed(116)
    np.random.shuffle(train_x)
    np.random.seed(116)
    np.random.shuffle(train_y)

    cnn_model = vgg13_model(input_shape=(224, 224, 3), classes=62)
    cnn_model.summary()

    # 设置TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 当验证集上的loss不再下降时就提前结束训练
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002,
                                                  patience=10, mode='auto')

    callbacks = [tensorboard_callback, early_stop]

    cnn_model.fit(train_x, train_y,
                  batch_size=100, epochs=300,
                  verbose=1, validation_split=0.2,
                  callbacks=callbacks)
