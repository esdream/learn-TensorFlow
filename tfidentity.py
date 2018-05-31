'''tf.identity()

tf.identity的作用是在计算图中添加一个新的节点，保证每次run一个Variable时，对应的operator都会执行。
'''

import tensorflow as tf

x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    # 如果这里是y = x，则会返回0, 0, 0, 0, 0
    y = tf.identity(x) # 返回1, 2, 3, 4, 5

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for i in range(5):
        print(y.eval())