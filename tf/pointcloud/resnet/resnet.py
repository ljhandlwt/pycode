from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.python.keras.layers import Conv2D,Dense,MaxPooling2D,ZeroPadding2D,Flatten,BatchNormalization,Dropout,AveragePooling2D,Activation
from tensorflow.contrib.keras.python.keras.layers import Conv3D,MaxPooling3D,AveragePooling3D,ZeroPadding3D
from tensorflow.contrib.keras.python.keras import layers
import logging
import h5py
import pdb

import utils


logger = utils.make_logger('resnet', is_stdout=False, filename='./log/resnet.log')

class ResNet(object):
    def __init__(self, args):
        self.args = args

        self.init_input()
        self.init_model()
        self.init_loss()

    def init_input(self):
        self.input_img = tf.placeholder(tf.uint8, [None,230,102,20,1])
        self.input_lr = tf.placeholder(tf.float32)
        self.input_label = tf.placeholder(tf.int32, [None])

        logger.info("input inited!")

    def init_model(self):
        with tf.variable_scope('resnet'):
            x = self.input_img
            x = tf.cast(x, tf.float32) / 255

            #split (1,2,3)
            x = self.split_block(x, 's1')

            #4
            x = self.conv_block(x, 3, [32,32,128], stage=4, block='a')
            x = self.identity_block(x, 3, [32,32,128], stage=4, block='b')
            x = self.identity_block(x, 3, [32,32,128], stage=4, block='c')
            x = self.identity_block(x, 3, [32,32,128], stage=4, block='d')
            x = self.identity_block(x, 3, [32,32,128], stage=4, block='e')
            x = self.identity_block(x, 3, [32,32,128], stage=4, block='f')

            #5
            x = self.conv_block(x, 3, [64,64,256], stage=5, block='a')
            x = self.identity_block(x, 3, [64,64,256], stage=5, block='b')
            x = self.identity_block(x, 3, [64,64,256], stage=5, block='c')

            #pool
            x = AveragePooling3D((7,4,2), name='avg_pool')(x)
            x = Flatten()(x)

            self.output_feature = x
            logger.info('feature shape:{}'.format(self.output_feature.shape))

            #fc
            x = Dense(self.args.classes, activation='softmax', name='fc')(x)

            self.output = x

        logger.info('network inited!')

    def split_block(self, x, split_name):
        split_name = '_' + split_name

        ##x = ZeroPadding2D((3,3))(x)
        x = Conv3D(16, (7,3,3), strides=(2,2,1), name='conv1'+split_name)(x)
        x = BatchNormalization(axis=4, name='bn_conv1'+split_name)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((3,2,2), strides=(2,2,2))(x)

        x = self.conv_block(x, 3, [8,8,32], stage=2, block='a'+split_name, strides=(1,1,1))
        x = self.identity_block(x, 3, [8,8,32], stage=2, block='b'+split_name)
        x = self.identity_block(x, 3, [8,8,32], stage=2, block='c'+split_name)

        x = self.conv_block(x, 3, [16,16,64], stage=3, block='a'+split_name)
        x = self.identity_block(x, 3, [16,16,64], stage=3, block='b'+split_name)
        x = self.identity_block(x, 3, [16,16,64], stage=3, block='c'+split_name)
        x = self.identity_block(x, 3, [16,16,64], stage=3, block='d'+split_name)

        return x

    def init_loss(self):
        label_onehot = tf.one_hot(self.input_label, self.args.classes)
        cross_entropy = -tf.reduce_sum(label_onehot*tf.log(self.output+self.args.eps), reduction_indices=[1])

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regular_vars = [var for var in train_vars if var.name.find('kernel')!=-1]
        regularizers = tf.add_n([tf.nn.l2_loss(var) for var in regular_vars])

        self.loss = tf.reduce_mean(cross_entropy) + self.args.weight_decay * regularizers

        self.opt = tf.train.AdamOptimizer(self.input_lr)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.grad = tf.gradients(self.loss, variables)

        bn_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = [self.opt.apply_gradients(zip(self.grad,variables))] + bn_op

        corr_pred = tf.equal(tf.cast(self.input_label,tf.int64), tf.argmax(self.output,1))
        self.acc_num = tf.reduce_sum(tf.cast(corr_pred, tf.int32))

        logger.info('loss inited!')

    def load_model(self, sess, saver):
        if self.args.model_path is not None:
            saver.restore(sess, self.args.model_path)
            logger.info('load model by restoring training model')
        else:
            logger.warning('please load model by random init')

    def conv_block(self, input, kernel, filters, stage, block, strides=(2,2,2)):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = input
        x = Conv3D(filters[0], (1,1,1), strides=strides, name=conv_name_base+'2a')(x)
        x = BatchNormalization(axis=4, name=bn_name_base+'2a')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters[1], kernel, padding='same', name=conv_name_base+'2b')(x)
        x = BatchNormalization(axis=4, name=bn_name_base+'2b')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters[2], (1,1,1), name=conv_name_base+'2c')(x)
        x = BatchNormalization(axis=4, name=bn_name_base+'2c')(x)

        shortcut = input
        shortcut = Conv3D(filters[2], (1,1,1), strides=strides, name=conv_name_base+'1')(shortcut)
        shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def identity_block(self, input, kernel, filters, stage, block):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) +block + '_branch'

        x = input
        x = Conv3D(filters[0], (1,1,1), name=conv_name_base+'2a')(x)
        x = BatchNormalization(axis=4, name=bn_name_base+'2a')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters[1], kernel, padding='same', name=conv_name_base+'2b')(x)
        x = BatchNormalization(axis=4, name=bn_name_base+'2b')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters[2], (1,1,1), name=conv_name_base+'2c')(x)
        x = BatchNormalization(axis=4, name=bn_name_base+'2c')(x)

        x = layers.add([x,input])
        x = Activation('relu')(x)
        return x
