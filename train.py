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

MODEL_SAVE_PATH = "myModel2"
MODEL_NAME = "model"


def train():
    print('load data......')
    validData = dataset.getdata(BATCH_SIZE, "valid")
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
        ckpt = tf.train.get_checkpoint_state("myModel")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        model_train = [[model.add_global1, model.cost1, model.train_op1],
                       [model.add_global2, model.cost2, model.train_op2],
                       [model.add_global3, model.cost3, model.train_op3],
                       [model.add_global4, model.cost4, model.train_op4],
                       [model.add_global5, model.cost5, model.train_op5],
                       ]
        model_valid = [model.predict1, model.predict2, model.predict3, model.predict4, model.predict5]
        for i in range(1):
            nowacc = 0.0
            maxacc = 0.0
            tranacc = 0.0
            gstep = 0
            while True:
                trainData = dataset.getdata(BATCH_SIZE, "train")
                if len(trainData[0]) > 2350:
                    break
            for ijk in range(4):
                index = list(range(len(trainData[0])))
                random.shuffle(index)
                for j in index:
                    gstep, cost, _ = sess.run(model_train[i],
                                              feed_dict={
                                                  model.imge: trainData[0][j],
                                                  model.lable: trainData[1][j],
                                                  model.training: True
                                              })
                    if gstep % 100 == 1:
                        s = 'net %d. After %d steps, cost is %.5f, nowacc: %.5f, maxacc: %.5f. trainacc: %.5f.' % (
                            i, gstep, cost, nowacc, maxacc, tranacc)
                        #f = open('out.txt', 'w')
                        #f.write(s)
                        #f.close()
                        print(s)
                nowacc = val2(sess, model, validData, test_num, model_valid[i])
                if nowacc > maxacc:
                    maxacc = nowacc
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=gstep+i*3000)
                tranacc = val(sess, model, trainData, test_num, model_valid[i])
                #if gstep >= 10000:
                    #break


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
