
# coding: utf-8
import tensorflow as tf

# ### Nerual Network

# In[60]:

import numpy as np
import matplotlib.pyplot as plt
# 定义一个神经层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # random_normal初始化的值服从正态分布
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    
# 自己生成数据
# numpy.linspace在指定的间隔内返回均匀间隔的数字
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

layer_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer_1, 10, 1, activation_function=None)

# tf.reduce_sum中的axis参数,传入的是求和的维度,如果为0则为第一维度求和(去掉1层[]求和),如果为1则为第二维度求和(去掉2层[]求和),如果为None则将所有元素求和
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    # 绘制样本
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#         测试axis
#         if(i == 0):
#             print('axis=0: ', sess.run(tf.reduce_sum(tf.square(ys - prediction), axis=0), feed_dict={xs: x_data, ys: y_data}))
#             print('axis=1: ', sess.run(tf.reduce_sum(tf.square(ys - prediction), axis=1), feed_dict={xs: x_data, ys: y_data}))
#             print('axis=None: ', sess.run(tf.reduce_sum(tf.square(ys - prediction), axis=None), feed_dict={xs: x_data, ys: y_data}))
        if(i % 50 == 0):
#             print('loss: ', sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)


# In[ ]:



