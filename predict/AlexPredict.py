
import tensorflow as tf
import sys
sys.path.append("H:\labelImg-master")

import numpy as np
from nets.AlexNet import AlexNet
from classes import classes


"""
This python file is used for
predicting the class of cells using
the alexnet.

"""

MODEL_SAVE_PATH = 'H:\Cells\SIH\checkpoints\AlexNet'

img_size = 224

def predict(image,boolean=False):
    x = tf.placeholder(tf.float32,(1,img_size,img_size,3))

    model = AlexNet(x,1.0,25,None)

    score = model.fc8

    saver = tf.train.Saver()

    with tf.Session() as sess:
        #f = tf.gfile.FastGFile(filepath, 'rb').read()
        #image = tf.image.decode_jpeg(f, channels=3)
        image = tf.cast(image,tf.uint8)
        image = tf.image.resize_images(image, [img_size, img_size])

        if image.get_shape()[-1]!=3:
            return False,None,None

        image = tf.reshape(image,(1,224,224,3))

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        # 在这里面跑的很多函数都要带上sess才行
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            result = sess.run(score,feed_dict={x:image.eval(session=sess)})

            softmax = tf.nn.softmax(result).eval(session=sess)
            #notice that the argamx will reduce the dimension while the argsort will not

            first_prediction = np.argmax(softmax, 1)[0]
            first_class_name = classes[first_prediction]
            second_prediction = np.argsort(softmax)[0,-2]
            second_class_name = classes[second_prediction]
            third_prediction = np.argsort(softmax)[0,-3]
            third_class_name = classes[third_prediction]
            fourth_prediction = np.argsort(softmax)[0,-4]
            fourth_class_name = classes[fourth_prediction]
            fifth_prediction = np.argsort(softmax)[0,-5]
            fifth_class_name = classes[fifth_prediction]
            prob_list = [softmax[0, first_prediction], softmax[0, second_prediction], softmax[0, third_prediction],
                         softmax[0, fourth_prediction], softmax[0, fifth_prediction]]
            name_list = [first_class_name, second_class_name, third_class_name, fourth_class_name, fifth_class_name]

            if boolean:
                print('Top-5 classes and probabilities:')
                for i in range(5):
                    print(name_list[i],prob_list[i])
            else:
                print('The class of the image is:', first_class_name, 'The probability is:',
                      softmax[0, first_prediction])

            return True,name_list,prob_list


        else:
            print('No checkpoint file found.')
            return


if __name__ == '__main__':
    predict('H:\Cells\SIH\切割图片\单核细胞系统-单核细胞\单核细胞系统-单核细胞_SNAP-153200-0013_1_1new.jpg',True)

