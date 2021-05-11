import dataset
import Model as Model
import tensorflow as tf
import os
import nltk
import random

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
REGULARIZER = 0.0001
BATCH_SIZE = 100

MODEL_SAVE_PATH = ["inception", "mutiona_v1"]
MODEL_NAME = "model"


def train():
    print('load data......')
    validData = dataset.getdata(BATCH_SIZE, "test")
    bacth_num = 2400
    test_num = len(validData[0])
    print('load finish')
    print(bacth_num)
    print(test_num)
    initializer = tf.random_uniform_initializer(-0.01, 0.01)
    with tf.variable_scope('model', reuse=None, initializer=initializer):
        model = Model.Model(bacth_num)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.853)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state("myModel2")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('start')

        print(val2(sess, model, validData, test_num, model.predict5))


def val(sess, model, data, test_num, model_predict):
    acc_num = 0
    tot_num = 0
    for i in range(min(test_num, len(data[0]))):
        predic = sess.run(model_predict, feed_dict={
            model.imge: data[0][i],
            model.training: True})
        for j in range(len(predic)):
            if predic[j] == data[1][i][j]:
                acc_num += 1
            tot_num += 1
    return acc_num / tot_num


def check(predict):
    cut = 0
    for i in predict:
        if int(i) == 0:
            cut += 1
    if cut >= 3:
        return 0
    return 1


def val2(sess, model, data, test_num, model_predict):
    acc_num = 0
    tot_num = 0
    for i in range(test_num):
        predic = sess.run(model_predict, feed_dict={
            model.imge: data[0][i],
            model.training: True})
        for j in range(len(predic) // 5):
            if check(predic[j * 5:j * 5 + 5]) == data[1][i][j * 5]:
                acc_num += 1
            tot_num += 1
    return acc_num / tot_num


train()
