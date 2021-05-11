import tensorflow as tf
from tensorflow.contrib import *
import numpy as np
from tensorflow.contrib.slim import nets
import inception
import mution_v1
import mution_v2
import mution_v3
import mution_v4

slim = tf.contrib.slim

MAX_GRAD_NORM = 2


class Model(object):
    def __init__(self, bacth_num):
        self.bacth_num = bacth_num
        kernel_5x5 = np.array([
            [-1.0 / 12, 2.0 / 12, -2.0 / 12, 2.0 / 12, -1.0 / 12],
            [2.0 / 12, -6.0 / 12, 8.0 / 12, -6.0 / 12, 2.0 / 12],
            [-2.0 / 12, 8.0 / 12, -12.0 / 12, 8.0 / 12, -2.0 / 12],
            [2.0 / 12, -6.0 / 12, 8.0 / 12, -6.0 / 12, 2.0 / 12],
            [-1.0 / 12, 2.0 / 12, -2.0 / 12, 2.0 / 12, -1.0 / 12]
        ])
        kernel_5x5 = tf.convert_to_tensor(kernel_5x5, dtype=tf.float32)
        kernel_5x5 = tf.reshape(kernel_5x5, [5, 5, 1, 1])

        with tf.variable_scope("net"):
            with tf.variable_scope("Input"):
                self.imge = tf.placeholder(tf.float32, [None, 224, 224])
                self.lable = tf.placeholder(tf.int32, [None])
                self.training = tf.placeholder(tf.bool)
                imge = tf.to_float(self.imge, name='ToFloat')
                imge = (imge - 98.72) / 62.25
                imge = tf.reshape(imge, [-1, 224, 224, 1])
                imge1 = tf.nn.conv2d(imge, kernel_5x5, strides=[1, 1, 1, 1], padding='SAME')
                net = tf.concat([imge1, imge1, imge1], axis=-1)

            logits1 = self.get_logits(net, inception, "inception")
            self.train_op1, self.add_global1, self.predict1, self.cost1 = self.get_loss(logits1, "inception")

            logits2 = self.get_logits(net, mution_v1, "mution_v1")
            logits2 = logits2 + logits1
            self.train_op2, self.add_global2, self.predict2, self.cost2 = self.get_loss(logits2, "mution_v1")

            logits3 = self.get_logits(net, mution_v2, "mution_v2")
            logits3 = logits2 + logits3
            self.train_op3, self.add_global3, self.predict3, self.cost3 = self.get_loss(logits3, "mution_v2")

            logits4 = self.get_logits(net, mution_v3, "mution_v3")
            logits4 = logits4 + logits3
            self.train_op4, self.add_global4, self.predict4, self.cost4 = self.get_loss(logits4, "mution_v3")

            logits5 = self.get_logits(net, mution_v4, "mution_v4")
            logits5 = logits4 + logits5
            self.train_op5, self.add_global5, self.predict5, self.cost5 = self.get_loss(logits5, "mution_v4")

    def get_logits(self, net, mynets, scope):
        with tf.variable_scope(scope):
            with slim.arg_scope(mynets.inception_v1_arg_scope()):
                net, endpoints = mynets.inception_v1(net, num_classes=1001)
            net = endpoints['Mixed_5c']
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            net = tf.reshape(net, [-1, 1024])
            net = tf.nn.dropout(net, 0.8)
            logits = tf.layers.dense(net, 2, use_bias=True,
                                     kernel_initializer=tf.constant_initializer(0),
                                     bias_initializer=tf.constant_initializer(0))
            return logits

    def get_loss(self, logits, scope):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.lable, [-1]))
        cost = tf.reduce_mean(loss)
        global_step = tf.Variable(0, trainable=False)
        loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/net/" + scope)
        train_op = tf.train.AdamOptimizer(5e-5).minimize(cost, var_list=loss_vars)
        add_global = global_step.assign_add(1)
        predict = tf.argmax(tf.nn.softmax(logits), 1)
        return train_op, add_global, predict, cost
