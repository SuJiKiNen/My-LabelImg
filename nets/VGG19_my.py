import tensorflow as tf
import numpy as np

"""
This python file implements the VGG-19
network used for training.

"""

NUM_LABELS = 25

def get_weights(shape,regularizer):
    weights = tf.get_variable("weigths",shape=shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))

    return weights


def get_bias(shape):
    bias = tf.get_variable('bias',shape=shape,
                           initializer=tf.constant_initializer(stddev=0.0))
    return bias


def conv2d(input,filter_height,filter_width,
           filter_num,stride_y,stride_x,name,regularizer,padding='SAME'):

    input_channels = int(input.get_shape()[-1])

    with tf.variable_scope(name):
        weights = get_weights(shape=[filter_height,
                                    filter_width,
                                    input_channels,
                                    filter_num],regularizer=regularizer)
        bias = get_bias(shape=[filter_num])

    conv = tf.nn.conv2d(input=input,filter=weights,
                        strides=[1,stride_y,stride_x,1],padding=padding)

    relu = tf.nn.relu(tf.add(conv,bias))
    return relu


def max_pool(input,filter_height,filter_width,
                stride_y,stride_x,name,padding='SAME'):
    return tf.nn.max_pool(input=input,
                          ksize=[1,filter_height,filter_width,1],strides=[1,stride_y,stride_x,1],padding=padding)

def fc(input,num_in,num_out,name,regularizeri,relu=True):
    with tf.variable_scope(name):
        weights = get_weights(shape=[num_in,num_out],regularizer=regularizer)
        bias = get_bias(shape=[num_out])

    fc = tf.nn.bias_add(tf.matmul(input,weights),bias)
    
    if relu:
        fc = tf.nn.relu(relu)

    return fc

def dropout(x,keep_prob):
    return tf.nn.dropout(x,keep_prob=keep_prob)


def VGG19(x,regularizer,keep_prob):
    conv1 = conv2d(x,3,3,64,1,1,'layer1-conv1',regularizer)
    conv2 = conv2d(conv1,3,3,64,1,1,'layer2-conv2',regularizer)
    pool1 = max_pool(conv1,3,3,2,2,'layer3-pool1')

    conv3 = conv2d(pool1,3,3,128,1,1,'layer4-conv3',regularizer) 
    conv4 = conv2d(conv3,3,3,128,1,1,'layer5-conv4',regularizer) 
    pool2 = max_pool(conv4,3,3,2,2,'layer6-pool2')

    
    conv5 = conv2d(pool2,3,3,256,1,1,'layer7-conv5',regularizer) 
    conv6 = conv2d(conv5,3,3,256,1,1,'layer8-conv6',regularizer) 
    conv7 = conv2d(conv6,3,3,256,1,1,'layer9-conv7',regularizer) 
    conv8 = conv2d(conv7,3,3,256,1,1,'layer10-conv8',regularizer) 
    pool3 = max_pool(conv8,3,3,2,2,'layer11-pool3')

    conv9 = conv2d(pool3,3,3,512,1,1,'layer12-conv9',regularizer)
    conv10 = conv2d(conv9,3,3,512,1,1,'layer13-conv10',regularizer)
    conv11 = conv2d(conv10,3,3,512,1,1,'layer14-conv11',regularizer)
    conv12 = conv2d(conv11,3,3,512,1,1,'layer15-conv12',regularizer)
    pool4 = conv2d(conv12,3,3,2,2,'layer16-pool4')
    
    
    conv13 = conv2d(pool4,3,3,512,1,1,'layer17-conv13',regularizer)
    conv14 = conv2d(conv13,3,3,512,1,1,'layer18-conv14',regularizer)
    conv15 = conv2d(conv14,3,3,512,1,1,'layer19-conv15',regularizer)
    conv16 = conv2d(conv15,3,3,512,1,1,'layer20-conv16',regularizer)
    pool5 = conv2d(conv16,3,3,2,2,'layer21-pool5')
    
    fc1 = fc(pool5,512,4096,regularizer)
    dropout1 = dropout(fc1,keep_prob)

    fc2 = fc(dropout1,4096,4096,regularizer)
    dropout2 = dropout(fc2,keep_prob)
    
    fc3 = fc(dropout2,4096,1000,regularizer)
    dropout3 = dropout(fc3,keep_prob)

    ws = get_weights([1000,NUM_LABELS],regularizer=regularizer)
    bs = get_bias([NUM_LABELS])
    softmax = tf.nn.softmax(tf.matmul(dropout3,ws) + bs)

    return softmax


