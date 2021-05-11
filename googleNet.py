import tensorflow as tf
from tensorflow.contrib import *
import numpy as np
from tensorflow.contrib.slim import nets

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
                imge = (imge-98.72)/62.25
                # imge = tf.layers.batch_normalization(imge, training=self.training)
                imge = tf.reshape(imge, [-1, 224, 224, 1])
                imge1 = tf.nn.conv2d(imge, kernel_5x5, strides=[1, 1, 1, 1], padding='SAME')
                # net = tf.image.resize_images(images=imge1, size=(224, 224))
                net = tf.concat([imge1, imge1, imge1], axis=-1)
            with slim.arg_scope(nets.inception.inception_v1_arg_scope()):
                net, endpoints = nets.inception.inception_v1(net, num_classes=1001)

            net = endpoints['Mixed_5c']
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            net = tf.reshape(net, [-1, 1024])
            net = tf.nn.dropout(net, 0.8)
            logits = tf.layers.dense(net, 2, use_bias=True,
                                     kernel_initializer=tf.constant_initializer(0),
                                     bias_initializer=tf.constant_initializer(0))

            # logits = net

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.lable, [-1]))
            self.cost = tf.reduce_mean(loss)
            trainable_variables = tf.trainable_variables()

            grads = tf.gradients(self.cost, trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)

            global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(5e-5,
                                                            global_step=global_step,
                                                            decay_steps=self.bacth_num,
                                                            decay_rate=0.98,
                                                            staircase=True)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            #self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            self.add_global = global_step.assign_add(1)
            self.predict = tf.argmax(logits, 1)
