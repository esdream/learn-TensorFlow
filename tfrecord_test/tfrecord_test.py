import os
import sys
from PIL import Image
import tensorflow as tf
import numpy as np

# 存入tfrecords文件
classes = {'healthy_g_band'}
writer = tf.python_io.TFRecordWriter('face_one.tfrecords')
img = Image.open('1_g_band.jpg')

img_raw = img.tobytes()
example = tf.train.Example(features=tf.train.Features(feature={
    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
}))
writer.write(example.SerializeToString())
writer.close()

# 从文件名队列读取，在通过tf.train.start_queue_runners()启动之前，文件名队列处于阻塞状态
filename_queue = tf.train.string_input_producer(['face_one.tfrecords'])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example,
                                    features={
                                        'label': tf.FixedLenFeature([], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)
                                    })

image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [128, 128])
# image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
image = tf.cast(image, tf.float32)
label = tf.cast(features['label'], tf.int32)
# one_hot编码，将label转化成[0,0,0,0,1,0,0,0]格式
one_hot_labels = tf.to_float(tf.one_hot(indices=label, depth=2, on_value=1, off_value=0))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

coord = tf.train.Coordinator()
# 启动文件名队列
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

samples = sess.run(image)
labels = sess.run(label)

# 显示图片
img = Image.fromarray(samples, 'L')
img.show()

# 将主线程阻塞，待读取线程完成后再进行
coord.request_stop()
coord.join(threads=threads)
