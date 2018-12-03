import tensorflow as tf
import numpy as np

"""
The file is the implementation of
the VGG-16 neural network

"""

NUM_LABELS = 25

def get_weights(shape, regularizer):
    weights = tf.get_variable('weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_bias(shape):
    bias = tf.get_variable('bias', shape=shape, initializer=tf.constant_initializer(0.0))
    return bias


def conv2d(x, filters, strides):
    return tf.nn.conv2d(x, filters, strides, padding='SAME')

def max_pool(x, ksize, strides, padding='SAME'):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

def VGG16(x, regularizer,keep_prob):
    # 由于有多层，所以不用with定义层数而用for循环
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = get_weights([3, 3, 3, 64], regularizer=regularizer)
        conv1_x = conv2d(x, conv1_weights,[1, 1, 1, 1])
        conv1_bias = get_bias([64])
        conv1_relu = tf.nn.relu(tf.add(conv1_x, conv1_bias))

    with tf.variable_scope('layer2-pool1'):
        pool1_x = max_pool(conv1_relu, [1, 3, 3, 1], [1, 2, 2, 1])

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = get_weights([3, 3, 64, 128], regularizer=regularizer)
        conv2_x = conv2d(pool1_x, conv2_weights,[1, 1, 1, 1])
        conv2_bias = get_bias([128])
        conv2_relu = tf.nn.relu(tf.add(conv2_x, conv2_bias))

    with tf.variable_scope('layer4-pool2'):
        pool2_x = max_pool(conv2_relu, [1, 3, 3, 1], [1, 2, 2, 1])

    with tf.variable_scope('layer5-conv3'):
        conv3_weights = get_weights([3, 3, 128, 256], regularizer=regularizer)
        conv3_x = conv2d(pool2_x, conv3_weights,[1, 1, 1, 1])
        conv3_bias = get_bias([256])
        conv3_relu = tf.nn.relu(tf.add(conv3_x, conv3_bias))

    with tf.variable_scope('layer6-conv4'):
        conv4_weights = get_weights([3, 3, 256, 256], regularizer=regularizer)
        conv4_x = conv2d(conv3_relu, conv4_weights, strides = [1, 1, 1, 1])
        conv4_bias = get_bias([256])
        conv4_relu = tf.nn.relu(tf.add(conv4_x, conv4_bias))

    with tf.variable_scope('layer7-pool3'):
        pool3_x = max_pool(conv4_relu, [1, 3, 3, 1], [1, 2, 2, 1])

    with tf.variable_scope('layer8-conv5'):
        conv5_weights = get_weights([3, 3, 256, 512], regularizer=regularizer)
        conv5_x = conv2d(pool3_x, conv5_weights, [1, 1, 1, 1])
        conv5_bias = get_bias([512])
        conv5_relu = tf.nn.relu(tf.add(conv5_x, conv5_bias))

    with tf.variable_scope('layer9-conv6'):
        conv6_weights = get_weights([3, 3, 512, 512], regularizer=regularizer)
        conv6_x = conv2d(conv5_relu, conv6_weights, [1, 1, 1, 1])
        conv6_bias = get_bias([512])
        conv6_relu = tf.nn.relu(tf.add(conv6_x, conv6_bias))

    with tf.variable_scope('layer10-pool4'):
        pool5_x = max_pool(conv6_relu, [1, 3, 3, 1], [1, 2, 2, 1])

    with tf.variable_scope('layer11-conv7'):
        conv7_weights = get_weights([3, 3, 512, 512], regularizer=regularizer)
        conv7_x = conv2d(pool5_x, conv7_weights, [1, 1, 1, 1])
        conv7_bias = get_bias([512])
        conv7_relu = tf.nn.relu(tf.add(conv7_x, conv7_bias))

    with tf.variable_scope('layer12-conv8'):
        conv8_weights = get_weights([3, 3, 512, 512], regularizer=regularizer)
        conv8_x = conv2d(conv7_relu, conv8_weights, [1, 1, 1, 1])
        conv8_bias = get_bias([512])
        conv8_relu = tf.nn.relu(tf.add(conv8_x, conv8_bias))

    with tf.variable_scope('layer13-pool5'):
        pool5 = max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1])
        shape = pool5.get_shape()
        nodes = shape[1] * shape[2] * shape[3]
        pool5_x = tf.reshape(pool5, [shape[0], nodes])
        # keep_prob = tf.placeholder('float')

    with tf.variable_scope('layer14-fc1'):
        fc1_weights = get_weights(shape=[nodes, 4096], regularizer=regularizer)
        fc1_bias = get_bias([4096])
        fc1_relu = tf.nn.relu(tf.matmul(pool5_x, fc1_weights) + fc1_bias)
        #add the dropout regularization
        dropout1 = tf.nn.dropout(fc1_relu, keep_prob=keep_prob)

    with tf.variable_scope('layer15-fc2'):
        fc2_weights = get_weights(shape=[4096, 4096], regularizer=regularizer)
        fc2_bias = get_bias([4096])
        fc2_relu = tf.nn.relu(tf.matmul(dropout1, fc2_weights) + fc2_bias)
        #add the dropout regularization
        dropout2 = tf.nn.dropout(fc2_relu, keep_prob=keep_prob)

    with tf.variable_scope('layer16-fc3'):
        fc3_weights = get_weights(shape=[4096, 1000], regularizer=regularizer)
        fc3_bias = get_bias([1000])
        fc3_relu = tf.nn.relu(tf.matmul(dropout2, fc3_weights) + fc3_bias)
        #add the dropout regularization
        dropout3 = tf.nn.dropout(fc3_relu, keep_prob=keep_prob)

    ws = get_weights([1000, NUM_LABELS], regularizer=regularizer)
    bs = get_bias([NUM_LABELS])
    y = tf.nn.softmax(tf.matmul(dropout3, ws) + bs)

    return y
