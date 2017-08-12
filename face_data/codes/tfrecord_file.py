
# coding: utf-8

# ## TFRecords
# 
# ### 定义与写入
# ---
# **TFRecords**是TensorFlow内定标准格式，其是一种二进制格式文件。
# TFRecords文件包含了`tf.train.Example`协议内存块，协议内存块内包含了字段`features`(`tf.trian.Features`),`features`中包含一个`feature`字典参数，其中key为feature名，value为`tf.train.Feature`格式。Example中默认使用三种类型数据:
# 
#     + Int64List
#     + FloatList
#     + BytesList
# 
# 将数据写入example协议内存块后，将example通过`example.SerializeToString()`方法序列化为一个字符串，通过`tf.python_io.TFRcordWriter`写入到TFRecords文件。

# In[235]:

import os
import sys
from PIL import Image
import tensorflow as tf
import numpy as np


# In[236]:
for data_dir in ['train_face_data', 'test_face_data']:
    path = os.path.join(os.path.abspath('.'), data_dir)
    classes = ['ill', 'healthy']
    writer = tf.python_io.TFRecordWriter(os.path.join(path, data_dir + '.tfrecords'))


    # In[237]:

    for index, name in enumerate(classes):
        class_path = os.path.join(path, name)
        
        if(os.path.isdir(class_path)):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))

                writer.write(example.SerializeToString())
                
    writer.close()


# ### 使用队列读取
# ---
# + 读取`tfrecords`文件可以使用文件名队列的读取方式。shuffle参数控制是否打乱顺序，num_epochs参数控制文件读取重复次数
#   
#    ```
#    filename_queue = tf.train.string_input_producer([filename], shuffle=False, num_epochs=0)
#    ```
#    
# + 通过`tf.parse_single_example()`解析example,返回值是dict形式的features。`tf.FixedLenFeature`解析feature
#  
# + `tf.decode_raw()`按指定格式转换二进制文件
# + `tf.cast()`转换元素类型
# + `tf.train.batch()`将tf对象转化成batch传输至graph中，num_threads参数控制读取的线程数
# + `tf.one_hot()`是TensorFlow提供的独热编码方法，indices参数是label列表，depth参数是label中分类的种类数，on_value是为这一种类时depth列表的值，off_value为不为这一种类时depth列表的值。返回一个维度为`indice * depth`的tensor

# In[238]:

# def read_and_decode(filename):
#     # 根据文件名生成队列
#     filename_queue = tf.train.string_input_producer([filename]) #读入流中
    
#     reader = tf.TFRecordReader() #返回文件名和文件
#     _, serialized_example = reader.read(filename_queue)

#     features = tf.parse_single_example(serialized_example,
#                                       features={
#                                           'label': tf.FixedLenFeature([], tf.int64),
#                                           'img_raw': tf.FixedLenFeature([], tf.string),
#                                       }) #取出包含image和label的feature对象
    
#     image = tf.decode_raw(features['img_raw'], tf.uint8)
#     image = tf.reshape(image, [128, 128])

#     # 将image正规化
# #     image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
#     image = tf.cast(image, tf.float32)
#     label = tf.cast(features['label'], tf.int32)

#     # 按batch读入
# #     batch_size = 100 
# #     capacity = 3 * batch_size
# #     image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
#     # one_hot编码labels
#     one_hot_labels = tf.to_float(tf.one_hot(indices=label, depth=2, on_value=1, off_value=0))
    
#     return image, one_hot_labels


# # + 调用`tf.train.start_queue_runners()`后，文件名队列才会开始填充，否则会被阻塞
# # + `reader.read()`返回两个Tensor对象，第一个是文件名（为什么文件名也是Tensor格式？而且如何解析？），第二个是文件内容

# # In[245]:

# def main():
    
#     path = os.path.join(os.path.abspath('.'), 'save_g_band')
#     tfrecords_filename = os.path.join(path, 'face_train.tfrecords')
#     image, label = read_and_decode(tfrecords_filename)
    
#     sess = tf.InteractiveSession()
#     tf.global_variables_initializer().run()

#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
#     for i in range(5):
#         samples = sess.run(image)
#         labels = sess.run(label)
#         img = Image.fromarray(samples, 'F')
#         img.show()
#         img.save('./' + str(i) + str(labels) + '.jpg')#存下图片

#     coord.request_stop()
#     coord.join(threads)


# # In[246]:

# if(__name__ == '__main__'):
#     main()


# # In[ ]:



