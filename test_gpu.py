import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 4.0], shape=[3], name='b')

# 原来的写法是tf.device('/gpu:0')。但在官方的最新文档中写法如下：
with tf.device('/device:GPU:0'):
    c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

# 在正确安装CUDA和CUDNN的情况下，测试GPU是否能够使用可能出现的问题：
# 1. Ignoring visible gpu device (device: 0, name: GPU名称, pci bus id: 0000:01:00.0) with Cuda compute capability 2.1. The minimum required Cuda capability is 3.0.
# 这种情况是由于Tensorflow使用CUDA加速的最小算力为3.0，而本机GPU的算力仅有2.1，因此无法使用GPU加速。也就是说，即使显卡能够支持CUDA和CUDNN，也不一定能够使用Tensorflow GPU版本的加速（所以还是乖乖放到云上面跑吧）。
# GPU对CUDA的支持和GPU算力见https://developer.nvidia.com/cuda-gpus.