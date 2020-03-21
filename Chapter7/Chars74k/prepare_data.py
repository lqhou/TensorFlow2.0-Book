#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: lqhou
@file: prepare_data.py
@time: 2019/09/02
"""

from PIL import Image
import os


# 重新缩放原图
def image_saving(filename):
    # 读取图片
    im = Image.open(filename)
    width, height = im.size
    # 缩放原图的模式：高质量
    scaler = Image.ANTIALIAS
    new_width = 224
    new_height = 224
    # im = im.convert('L')
    im_resize = im.resize((new_width, new_height), scaler)
    return im_resize


def image_preprocessing(path):
    old_path = path

    save_path = os.path.join(os.path.abspath(os.path.join(path, os.path.pardir)), 'chars74k_data')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    classesName = os.listdir(old_path)

    for i in classesName:
        file_path = os.path.join(old_path, i)
        savefiledir_path = os.path.join(save_path, i)

        filenames = os.listdir(file_path)

        if not os.path.exists(savefiledir_path):
            os.makedirs(savefiledir_path)

        for filename in filenames:
            # 读取每个文件夹下的原始图片
            image_read_path = os.path.join(file_path, filename)
            image_save_path = os.path.join(savefiledir_path, filename)
            # 将尺寸为1200*900的三通道RGB的图片转成了24*18的单通道灰度的图像
            image = image_saving(image_read_path)
            # 将处理过的图片保存在新的路径中
            image.save(image_save_path)

image_preprocessing('./BMP')
