import os

import numpy as np
import tensorflow as tf
import cv2

from SIH.nets.AlexNet import AlexNet
from SIH.utils.Alex_dataGenerate import ImageDataGenerator
from datetime import datetime
from SIH.classes import classes

#for plotting and displaying
import matplotlib.pyplot as plt

#compute the number of each class in the validation data
from SIH.compute_examples import compute_examples

"""
This python file is downloaded from github, finishing
the training process for the AlexNet using our own dataset.

"""
plt.rcParams['font.sans-serif'] = ['simhei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


"""
Configuration Part.
"""
#def iter_alex(train_file,val_file,batch_size,learning_rate,dropout_rate,img_size,count):

# Path to the textfiles for the trainings and validation set
train_file = '/home/dreamboy/Cells/SIH/trainData.txt'
val_file = '/home/dreamboy/Cells/SIH/validateData.txt'

# Learning params
learning_rate = 0.001
num_epochs = 50
batch_size = 16
img_size=224

# Network params
dropout_rate = 0.5
num_classes = 25
train_layers = ['fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/home/dreamboy/Cells/SIH/tensorboard/AlexNet"
checkpoint_path = "/home/dreamboy/Cells/SIH/checkpoints/AlexNet"

MODEL_SAVE_PATH = '/home/dreamboy/Cells/SIH/models'
MODEL_NAME = 'AlexNet.ckpt'

epochs = []
accuracies = []
counts = []
validate_nums = compute_examples(val_file)

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                    mode='training',
                                    batch_size=batch_size,
                                    num_classes=num_classes,
                                    img_size=img_size,
                                    shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                    mode='inference',
                                    batch_size=batch_size,
                                    num_classes=num_classes,
                                    img_size=img_size,
                                    shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, img_size, img_size, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                      labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))


    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
#for gradient, var in gradients:
    #tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
#for var in var_list:
    #tf.summary.histogram(var.name, var)

# Add the loss to summary
#tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    len = score.get_shape()[0]
    sess = tf.Session()

    """
    for i in range(len):
        if tf.argmax(score,1)[i] != tf.argmax(y,1)[i]:
            print('Label is' + classes[tf.argmax(y,1).eval(session=sess)[i]])
            print('Predicted label is:' + classes[tf.argmax(score,1)].eval(session=sess)[i])
            plt.imshow(x[i].eval(session=sess))
    """


    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))

#For the convenience of fine-tuning
print('data_size',tr_data.data_size)
print('batch_size:',batch_size)
print('epoches:',num_epochs)
print('train_batches_per_epoch',train_batches_per_epoch)

val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

    error_mat = np.zeros((num_classes, num_classes))
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        epochs.append(epoch)

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                            y: label_batch,
                                            keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0

        #count = 0

        if epoch == 49:
            class_count = 0
            num = 0
            for _ in range(val_batches_per_epoch):

                img_batch, label_batch = sess.run(next_batch)
                acc,score_ = sess.run([accuracy,score], feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                for i in range(label_batch.shape[0]):
                    actual_label = np.argmax(label_batch,1)[i]
                    predict_label = np.argmax(score_,1)[1]
                    num += 1
                    if predict_label != actual_label:
                        #print('Label is: ' + classes[actual_label])
                        #print('Predicted label is: ' + classes[predict_label])
                        img_rgb = cv2.cvtColor(img_batch[i],cv2.COLOR_BGR2RGB) + [123.68, 116.779, 103.939]
                        plt.imshow(img_rgb/255)
                        plt.title('predict:' + str(classes[predict_label]) + '\nactual:' + str(classes[actual_label]))
                        plt.axis('off')
                        plt.savefig('../error_predict/' + str(class_count) + '_' + str(predict_label) + '_' + str(actual_label) + '.png')
                        class_count = class_count + 1
                        error_mat[actual_label][predict_label] = error_mat[actual_label][predict_label] + 1
                        #count = count + 1

                print("The acc:",acc)
                test_acc += acc
                test_count += 1
            test_acc /= test_count

            print('class count:',class_count)
            print('num',num)

        #generate the error count list
        #counts.append(count)

        accuracies.append(test_acc)
        print("{} Validation Accuracy = {:.5f}".format(datetime.now(),
                                                           test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))

    print(error_mat)

    #absolute
    plt.figure(figsize=(15,8))
    plt.bar(range(0,25),np.sum(error_mat,1))
    for x,y in zip(range(0,25),np.sum(error_mat,1)):
        plt.text(x,y,(x,y),ha='center',va='bottom',fontsize=9)
    plt.xlabel('classes')
    plt.ylabel('errors')
    plt.title('The number of prediction errors of every class')
    plt.savefig('error_predict_per_class.png')

    #relative
    plt.figure(figsize=(15,8))
    array = np.sum(error_mat,1)
    print(array)
    for i in range(25):
        if validate_nums[i] == 0:
            continue
        array[i]/=validate_nums[i]
    plt.bar(range(0,25), array)
    for x,y in zip(range(0,25),array):
        plt.text(x,y,(x,'%.4f'%y),ha='center',va='bottom',fontsize=9)
    plt.xlabel('classes')
    plt.ylabel('errors_rate')
    plt.title('The relative comparision of prediction errors of every class')
    plt.savefig('error_predict_per_class_rate.png')


    #plotting the pie picture for the max_error_count class
    plt.figure()
    max_error_index = np.argmax(np.sum(error_mat,1))
    max_error_label = classes[max_error_index]

    sorted = np.argsort(error_mat[max_error_index,:])
    first_label = classes[sorted[-1]]
    first_percent = error_mat[max_error_index,sorted[-1]]/np.sum(error_mat,1)[max_error_index]

    second_label = classes[sorted[-2]]
    second_percent = error_mat[max_error_index, sorted[-2]] / np.sum(error_mat, 1)[max_error_index]

    third_label = classes[sorted[-3]]
    third_percent = error_mat[max_error_index, sorted[-3]] / np.sum(error_mat, 1)[max_error_index]


    labels_list = [first_label,second_label,third_label]

    percent_list = [first_percent,second_percent,third_percent]
    print(percent_list)

    plt.axes(aspect=1)
    print(labels_list,percent_list)
    plt.pie(x=percent_list, labels=labels_list,autopct='%3.1f %%',
    shadow=True, labeldistance=1.1, startangle = 90,pctdistance = 0.6)

    plt.title('The incorrectly predicted labels of ' + max_error_label)
    plt.show()


    """
    #plotting the accuracy figure
    plt.figure()
    plt.plot(epochs, accuracies)
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    # find the max accuracy:
    max_accuracy = accuracies.index(max(accuracies))
    x, y = epochs[max_accuracy], accuracies[max_accuracy]
    plt.text(x, y, '(%d,%.3f)' % (x, y), ha='center', va='bottom', fontsize=8)
    plt.title('epoches-accuracy')
    plt.savefig('epoches-accuracy' + '.png')

    #plotting the error counts
    plt.figure()
    plt.plot(range(1,51),counts)
    plt.xlabel('epoches')
    plt.ylabel('numbers')

    min_counts = counts.index(min(counts))
    x,y = min_counts,counts[min_counts]
    plt.text(x,y,'(%d,%d)' % (x, y), ha='center', va='bottom', fontsize=8)

    plt.title('error of the predictions')
    plt.savefig('error_predict.png')
    plt.show()
    """