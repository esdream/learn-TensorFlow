
# coding: utf-8

# In[3]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[4]:

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128 

n_inputs = 28 # 每一次input一行pixel,即n_inputs = 28
n_steps = 28 # 一共28列,因此n_steps = 28
n_hidden_units = 128 # neurons in hidden layer
n_classes = 10


# In[5]:

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights and biases
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# In[6]:

def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X(128 batch, 28 steps, 28 inputs) ==> X(128 * 28, 28 inputs)
    # 将X展成一个长度为28的向量,一共128 * 28个(reshape中第一个参数为-1,表示个数由总数除以shape得到)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in : (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    with tf.variable_scope('lstm', reuse=None):
         # LSTM cell
        # 在旧版本中lstm_cell = tf.nn.rnn_cell_BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tupe=True)
        # 其中forget_bias表示遗忘系数，等于1时不会遗忘任何信息。
        # state_is_tuple默认为True，官方建议为True。lstm cell会范围2个元素的元组(c_state, m_state)。
        # 其中c_state是主线state，m_state是分线state。这个属性马上将被启用，即只返回Tuple.
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        
        # output cell
        # dynamic RNN比传统RNN具有更好的效果
        # 其中time_major参数是inputs和outputs Tensor的形式。为True时，
        # Tensor的格式为[max_time, batch_size, depth]，为False时格式为
        #[batch_size, max_time, depth]。其中max_time是RNN的循环次数。
        # 官方文档上说time_major=True在计算时会更有效率因为避免了在
        # RNN开始和结束时的转置。但绝大部分Tensorflow数据是batch-major的。
        # 所以默认值为False
        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False, scope='lstm')
        
        # outputs是一个3-D Tensor，如果time_major==True，则为[max_time, batch_size, cell.output_size]
        # 如果time_major==False，则为[batch_size, max_time, cell.output_size]
        # 将outputs维度进行变换，[0, 1, 2]转化为[1, 0, 2]
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        # hidden layer for output as the final results
        # shape = (128, 10)
        # outputs[-1] = final_state[1] = c_state，也就是所有分线记忆输出。注意一下其他案例可能等式不成立。
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    
        return results


# In[7]:

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# tf.argmax此函数对矩阵按行或列计算最大值，0表示列，1表示行
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# tf.cast()将矩阵元素转换成需要的类型
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[8]:

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys
            }))
        step += 1

