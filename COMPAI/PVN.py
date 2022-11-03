#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import summary_ops_v2
import os


class policy_value_network(object):
    def __init__(self, learning_rate_fn, res_block_nums = 7):
        #         self.ckpt = os.path.join(os.getcwd(), 'models/best_model.ckpt-13999')    # TODO
        self.save_dir = "./models"
        self.is_logging = True

        if tf.io.gfile.exists(self.save_dir):
            #             print('Removing existing model dir: {}'.format(MODEL_DIR))
            #             tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(self.save_dir)

        train_dir = os.path.join(self.save_dir, 'summaries', 'train')
        test_dir = os.path.join(self.save_dir, 'summaries', 'eval')

        self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
        self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

        # Variables
        self.filters_size = 128    # or 256
        self.prob_size = 2086
        self.digest = None

        self.inputs_ = tf.keras.layers.Input([9, 10, 14], dtype='float32', name='inputs')  # TODO C plain x 2
        self.c_l2 = 0.0001
        self.momentum = 0.9
        self.global_norm = 100

        self.layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, padding='same')(self.inputs_)
        self.layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(self.layer)
        self.layer = tf.keras.layers.ReLU()(self.layer)

        # residual_block
        with tf.name_scope("residual_block"):
            for _ in range(res_block_nums):
                self.layer = self.residual_block(self.layer)

        # policy_head
        with tf.name_scope("policy_head"):
            self.policy_head = tf.keras.layers.Conv2D(filters=2, kernel_size=1, padding='same')(self.layer)
            self.policy_head = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(self.policy_head)
            self.policy_head = tf.keras.layers.ReLU()(self.policy_head)

            self.policy_head = tf.keras.layers.Reshape([9 * 10 * 2])(self.policy_head)
            self.policy_head = tf.keras.layers.Dense(self.prob_size)(self.policy_head)

        # value_head
        with tf.name_scope("value_head"):
            self.value_head = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(self.layer)
            self.value_head = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(
                self.value_head)
            self.value_head = tf.keras.layers.ReLU()(self.value_head)

            self.value_head = tf.keras.layers.Reshape([9 * 10 * 1])(self.value_head)
            self.value_head = tf.keras.layers.Dense(256, activation='relu')(self.value_head)
            self.value_head = tf.keras.layers.Dense(1, activation='tanh')(self.value_head)