"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import SIH.nets.vgg19_trainable as vgg19
from SIH.dataGenerate import data_generate
from SIH.utils import vgg_utils

IMG_WIDTH=224
IMG_HEIGHT=224

BATCH_SIZE=128
CAPACITY=256

INPUT_CHANNELS = 3
NUM_CLASSES = 25

TRAIN_STEPS = 30000

MODEL_PATH='/home/dreamboy/Cells/SIH/models/vgg19.npy'

images = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS])
true_out = tf.placeholder(tf.float32, [BATCH_SIZE])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19(MODEL_PATH)
vgg.build(images, train_mode)

cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

correction = tf.equal(tf.arg_max(vgg.prob, 1), tf.arg_max(true_out, 1))
accuracy = tf.reduce_mean(tf.cast(correction, 'float'))

with tf.device('/cpu:0'):
    with tf.Session() as sess:

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print(vgg.get_var_count())

        sess.run(tf.global_variables_initializer())

        # test classification
        #prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        #vgg_utils.print_prob(prob[0], './synset.txt')
        coord = tf.train.Coordinator()
        # start the queue,otherwise the system will be blocked

        image_batch_tensor, label_batch_tensor = data_generate(False, 'train', IMG_HEIGHT, IMG_HEIGHT, BATCH_SIZE, CAPACITY)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            i = 0
            while not coord.should_stop() and i < TRAIN_STEPS:
                # simple 1-step training

                image_batch, label_batch = sess.run([image_batch_tensor,label_batch_tensor])
                print(image_batch)
                _,loss = sess.run([train,cost],feed_dict={images:image_batch,true_out:label_batch,train_mode:True})
                print('After %d steps,the loss is: %g' % (i+1,loss))
                i = i + 1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        # stop
        coord.join(threads)

            #if i % 1000 == 0:
                #val_batch,val_label=data_generate(False,'validate',IMG_HEIGHT,IMG_WIDTH,BATCH_SIZE,CAPACITY)
                #accu = sess.run(accuracy,feed_dict={images:val_batch,true_out:val_label,train_mode:False})
                #print('After %d steps, the accuracy is: %g' % (i,accu))

            # test classification again, should have a higher probability about tiger
            #　prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
            #　vgg_utils.print_prob(prob[0], './synset.txt')

        # test save
        vgg.save_npy(sess, './test-save.npy')
