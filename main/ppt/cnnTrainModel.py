import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import time

import ppt.dataProduce as Producer

ABSPATH_DIR_NAME = os.path.abspath(os.path.dirname(__file__))
CAPTCHA_IMAGE_PATH = Producer.CAPTCHA_IMAGE_PATH
CAPTCHA_IMAGE_WIDTH = 160
CAPTCHA_IMAGE_HEIGHT = 60
CHAR_SET_LEN = Producer.CHAR_SET_LEN
CAPTCHA_LEN = Producer.CAPTCHA_LEN

TRAIN_IMAGE_PERCENT = 0.6
TRAIN_NAME_LIST = []
VALID_NAME_LIST = []

MODEL_SAVE_PATH = ABSPATH_DIR_NAME


def get_image_name(image_path=CAPTCHA_IMAGE_PATH):
    file_name = []
    for path in os.listdir(image_path):
        image_name = path.split('/')[-1]
        file_name.append(image_name)
    return file_name, len(file_name)


def get_one_image_label(name):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i, j in enumerate(name):
        index = i * CHAR_SET_LEN + ord(j) - ord('0')
        label[index] = 1
    return label


def get_data_and_label(image_name, file_path=CAPTCHA_IMAGE_PATH):
    image_path = os.path.join(file_path, image_name)
    img = Image.open(image_path)
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten() / 255
    image_label = get_one_image_label(image_name[:CAPTCHA_LEN])
    return image_data, image_label


def get_next_batch(batch_size=32, train_or_test='train', step=0):
    batch_data = np.zeros([batch_size, CAPTCHA_IMAGE_WIDTH * CAPTCHA_IMAGE_HEIGHT])
    batch_label = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])
    file_name_list = TRAIN_NAME_LIST
    if train_or_test == 'validate':
        file_name_list = VALID_NAME_LIST

    file_name_length = len(file_name_list)
    index_start = step * batch_size
    for i in range(batch_size):
        index = (i + index_start) % file_name_length
        name = file_name_list[index]
        img_data, img_label = get_data_and_label(name)
        batch_data[i, :] = img_data
        batch_label[i, :] = img_label

    return batch_data, batch_label


def weight_variable(shape, name='weight'):
    init = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布 shape张量维度，mean均值，stddev标准差
    var = tf.Variable(initial_value=init, name=name)  # 定义图变量
    return var


def bias_variable(shape, name='bias'):
    init = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial_value=init, name=name)
    return var


def conv2d(x, W, name='conv2d'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, name='maxpool'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def train():
    X = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_HEIGHT * CAPTCHA_IMAGE_WIDTH], name='data-input')
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name='label-input')
    x_input = tf.reshape(X, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDTH, 1], name='x-input')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')

    # first conv layer
    W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2x2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # second conv layer
    W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
    conv2 = max_pool_2x2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # third conv layer
    W_conv3 = weight_variable([5, 5, 64, 64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2x2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # full connected layer
    W_fc1 = weight_variable([20 * 8 * 64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    print(20 * 8 * 64,  W_fc1.get_shape().as_list()[0])
    fc1 = tf.reshape(conv3, [-1, 20 * 8 * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # output layer
    W_fc2 = weight_variable([1024, CAPTCHA_LEN * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([CAPTCHA_LEN * CHAR_SET_LEN], 'B_fc2')
    output = tf.nn.softmax(tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output'))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='labels')

    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        for epoch in range(6000):
            train_data, train_label = get_next_batch(256, 'train', steps)
            sess.run([optimizer, labels_max_idx], feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})
            if steps % 1 == 0:
                test_data, test_label = get_next_batch(10, 'validate', steps)
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob:1.0})
                print("steps: %d, acc: %f" % (steps, acc))

                if acc >= 0.95:
                    break
            steps += 1
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, "crack_captcha.model"), global_step=steps)


if __name__ == '__main__':
    image_file_name_list, num_of_image_file = get_image_name()
    random.seed(time.time())
    random.shuffle(image_file_name_list)

    num_of_train_image = int(num_of_image_file * TRAIN_IMAGE_PERCENT)
    TRAIN_NAME_LIST = image_file_name_list[:num_of_train_image]
    VALID_NAME_LIST = image_file_name_list[num_of_train_image:]

    train()
    print("End!")

