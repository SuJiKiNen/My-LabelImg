import tensorflow as tf
import numpy as np

"""
This python file implements
the LeNet-5 neural network for training
"""

#some parameters for the LeNet-5

INPUT_SIZE = 16900
#NODES_OUTPUT = 10

IMAGE_SIZE = 130
NUM_CHANNELS = 3
NUM_LABELS = 25

CONV1_SIZE = 5
CONV1_DEPTH = 32

CONV2_SIZE = 5
CONV2_DEPTH = 64

FC_SIZE = 512


#return the weights
def get_weights(shape,regularizer):
    weights = tf.get_variable('weights',shape=shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))

    return weights

#return the bias
def get_bias(shape):
    bias = tf.get_variable('bias',shape=shape,
                           initializer=tf.constant_initializer(0.0))
    return bias


def Lenet(x_input,train,regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = get_weights(shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEPTH],
                                    regularizer=regularizer)
        conv1 = tf.nn.conv2d(input=x_input,filter=conv1_weights,strides=[1,1,1,1],padding='SAME')
        bias = get_bias([CONV1_DEPTH])

        relu1 = tf.nn.relu(tf.add(conv1,bias))

    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = get_weights([CONV2_SIZE, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH], regularizer)
        conv2_bias = get_bias([CONV2_DEPTH])

        conv2 = tf.nn.conv2d(pool1, conv2_weights, [1, 1, 1, 1], padding='SAME')

        relu2 = tf.nn.relu(tf.add(conv2, conv2_bias))

    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool2_shape = pool2.get_shape()
    nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    pool2_reshape = tf.reshape(pool2, [pool2_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = get_weights([nodes,FC_SIZE],regularizer=regularizer)

        fc1_bias = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.matmul(pool2_reshape, fc1_weights) + fc1_bias

        if train:
            #add the dropout
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = get_weights([FC_SIZE,NUM_LABELS],regularizer=regularizer)

        fc2_bias = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights) + fc2_bias
        return logit