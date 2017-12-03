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
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    
    # 将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的
    distorted_image = tf.image.resize_images(distorted_image, height, width, method=np.random.randint(4))
    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np,random.randint(2))
    
    return distorted_image

if(__name__ == '__main__'):
    image_raw_data = tf.gfile.FastGFile('./image/cat.jpg', 'rb').read()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)

        for i in range(6):
            # 将图像尺寸调整为128, 128
            result = preprocess_for_train(img_data, 128, 128, None)
            plt.imshow(result.eval())
            plt.show()
            