import tensorflow as tf
import numpy as np
from SIH.nets.VGG16_my import VGG16
from SIH.nets.VGG19_my import VGG19
from SIH.nets.LeNet5 import Lenet

from SIH.dataGenerate import data_generate

from tensorflow.examples.tutorials.mnist import input_data

"""
This python file implements the training process
of the VGG16 or VGG19 network

"""

#some hyperparameters

BATCH_SIZE = 128
NUM_INPUT = 3
NUM_LABLES = 25

CAPACITY = 256
IMG_W = 130
IMG_H = 130

REGULARIZATION_RATE = 0.001
MOVING_AVERAGE_DECAY = 0.99

LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
MAX_STEPS = 30000

MODEL_SAVE_PATH = '../models/'

def train(type):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,IMG_H,IMG_W,NUM_INPUT])
    y = tf.placeholder(tf.float32,
                       [None,NUM_LABLES])
    keep_prob = tf.placeholder(tf.float32)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    if type == 'VGG16':
        y_ = VGG16(x,regularizer=regularizer,keep_prob=keep_prob)
    elif type == 'VGG19':
        y_ = VGG19(x,regularizer=regularizer,keep_prob=keep_prob)
    elif type == 'Lenet':
        y_ = Lenet(x,True,regularizer=regularizer)
    else:
        print('The type is wrong!')
        return

    variables_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_average_op = variables_average.apply(tf.trainable_variables())

    global_step = tf.Variable(0,trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_,labels=tf.argmax(y,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    correction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correction,'float'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)

    with tf.control_dependencies([train_step,variables_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        for i in range(MAX_STEPS):

            #convert the returned tensors, because tensors can not be feeded into the feed_dict
            xs,ys = data_generate(False,'train',IMG_H,IMG_W,BATCH_SIZE,CAPACITY)
            xs_reshape = xs.reshape([BATCH_SIZE,
                                        IMG_H,
                                        IMG_W,
                                        NUM_INPUT])

            _,step,loss_val = sess.run([train_op,global_step,loss],feed_dict={x:xs_reshape,y:ys,keep_prob:1.0})
            print('the loss value is:',loss_val)

            if step % 1000 == 0 :
                print('After %s training steps, the loss of the batch is: %g' % (step,loss_val))
                saver.save(sess=sess,save_path=MODEL_SAVE_PATH + type)


#For testing
if __name__ == '__main__':
    train('VGG16')
