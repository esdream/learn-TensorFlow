'''Image Preprocessing

Preprocess the image.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 不同的调整顺序会产生不同结果，可以自定义多种顺序。调用时随机选择一种。
def distort_color(image, color_ordering=0):
    if(color_ordering == 0):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif(color_ordering == 1):
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    # 如果没有提供标注框，则整个图像都是需要关注的部分
    if(bbox is None):
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    # 转换图像Tensor类型
    if(image.dtype != tf.float32):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像，减小需要关注的物体大小对图像识别算法的影响
