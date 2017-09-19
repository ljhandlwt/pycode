import inspect
import os

import numpy as np
import tensorflow as tf
import time

avgrageImage = 169.5361 / 255
FRAME_PER_CLIP = 8
CROPSIZE = 224

class_num = 501

num_gpu = 4

'''
jh:2017.8.31
multi-gpu

implement:
1. a vgg model with multi single model
   a single model represent a gpu
   so var defined on vgg model while ops defined on single model
2. all grads from single models gather to a avg grad and update by avg grad

note:
1. it's easy to crash if tot_size%(batch_size*num_gpu) != 0
'''

def make_conv(x, W, b, activation=tf.nn.relu):
    return activation(tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME') + b)

def make_fc(x, W, b, activation=tf.nn.relu):
    return activation(tf.matmul(x, W)+b)

def make_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def make_drop(x, kp):
    return tf.nn.dropout(x, keep_prob=kp)

def make_flat(x):
    shape = x.get_shape().as_list()
    return tf.reshape(x, [-1,shape[1]*shape[2]*shape[3]])        


class Single_GPU_MODEL:
    def __init__(self, parent):
        self.parent = parent

        self.model_setup()

    def model_setup(self):
        # input
        self.flow = tf.placeholder(tf.float32, [None, CROPSIZE, CROPSIZE, FRAME_PER_CLIP], name='input_data')
        self.labels = tf.placeholder(tf.int32, [None], name='input_labels')
        self.labels_one_hot = tf.placeholder(tf.float32, [None, class_num], name='data_labels_one_hot')
        # img scale
        data_in = self.flow - avgrageImage
        assert data_in.get_shape().as_list()[1:] == [CROPSIZE, CROPSIZE, FRAME_PER_CLIP]
        # net
        x = data_in
        x = make_conv(x, self.parent.conv1_1_W, self.parent.conv1_1_b)
        x = make_conv(x, self.parent.conv1_2_W, self.parent.conv1_2_b)
        x = make_pool(x)
        x = make_conv(x, self.parent.conv2_1_W, self.parent.conv2_1_b)
        x = make_conv(x, self.parent.conv2_2_W, self.parent.conv2_2_b)
        x = make_pool(x)
        x = make_conv(x, self.parent.conv3_1_W, self.parent.conv3_1_b)
        x = make_conv(x, self.parent.conv3_2_W, self.parent.conv3_2_b)
        x = make_conv(x, self.parent.conv3_3_W, self.parent.conv3_3_b)
        x = make_pool(x)
        x = make_conv(x, self.parent.conv4_1_W, self.parent.conv4_1_b)
        x = make_conv(x, self.parent.conv4_2_W, self.parent.conv4_2_b)
        x = make_conv(x, self.parent.conv4_3_W, self.parent.conv4_3_b)
        x = make_pool(x)
        x = make_conv(x, self.parent.conv5_1_W, self.parent.conv5_1_b)
        x = make_conv(x, self.parent.conv5_2_W, self.parent.conv5_2_b)
        x = make_conv(x, self.parent.conv5_3_W, self.parent.conv5_3_b)
        x = make_pool(x)
        x = make_flat(x)
        x = make_fc(x, self.parent.fc6_W, self.parent.fc6_b)
        x = make_drop(x, self.parent.keep_prob)
        x = make_fc(x, self.parent.fc7_W, self.parent.fc7_b)
        x = make_drop(x, self.parent.keep_prob)
        x = make_fc(x, self.parent.fc8_W, self.parent.fc8_b, activation=tf.identity)
        # output
        with tf.name_scope('logits'):
            self.logits = x
        with tf.name_scope('prob'):
            self.prob = tf.nn.softmax(x)
        # loss
        with tf.name_scope('loss'):
            cross_entropy = -tf.reduce_sum(self.labels_one_hot*tf.log(self.prob+1e-10), reduction_indices=[1])    

            regularizers = (tf.nn.l2_loss(self.parent.conv1_1_W) + tf.nn.l2_loss(self.parent.conv1_1_b) +
                            tf.nn.l2_loss(self.parent.conv1_2_W) + tf.nn.l2_loss(self.parent.conv1_2_b) +
                            tf.nn.l2_loss(self.parent.conv2_1_W) + tf.nn.l2_loss(self.parent.conv2_1_b) +
                            tf.nn.l2_loss(self.parent.conv2_2_W) + tf.nn.l2_loss(self.parent.conv2_2_b) +
                            tf.nn.l2_loss(self.parent.conv3_1_W) + tf.nn.l2_loss(self.parent.conv3_1_b) +
                            tf.nn.l2_loss(self.parent.conv3_2_W) + tf.nn.l2_loss(self.parent.conv3_2_b) +
                            tf.nn.l2_loss(self.parent.conv3_3_W) + tf.nn.l2_loss(self.parent.conv3_3_b) +
                            tf.nn.l2_loss(self.parent.conv4_1_W) + tf.nn.l2_loss(self.parent.conv4_1_b) +
                            tf.nn.l2_loss(self.parent.conv4_2_W) + tf.nn.l2_loss(self.parent.conv4_2_b) +
                            tf.nn.l2_loss(self.parent.conv4_3_W) + tf.nn.l2_loss(self.parent.conv4_3_b) +
                            tf.nn.l2_loss(self.parent.conv5_1_W) + tf.nn.l2_loss(self.parent.conv5_1_b) +
                            tf.nn.l2_loss(self.parent.conv5_2_W) + tf.nn.l2_loss(self.parent.conv5_2_b) +
                            tf.nn.l2_loss(self.parent.conv5_3_W) + tf.nn.l2_loss(self.parent.conv5_3_b))    

            self.loss = tf.reduce_mean(cross_entropy) + 0.0005*regularizers
        with tf.name_scope('grad'):
            self.grad_conv = tf.gradients(self.loss, self.parent.conv_var_list)
            self.grad_fc = tf.gradients(self.loss, self.parent.fc_var_list)
        # acc
        with tf.name_scope('evaluation'):
            self.correct_top_1 = tf.equal(tf.cast(self.labels,tf.int64), tf.argmax(self.logits,1))
            self.correct_top_5 = tf.nn.in_top_k(self.logits, self.labels, 5)

            self.accury_top_1 = tf.reduce_mean(tf.cast(self.correct_top_1, tf.float32))
            self.accury_top_5 = tf.reduce_mean(tf.cast(self.correct_top_5, tf.float32))

            self.correct_num_top_1 = tf.reduce_sum(tf.cast(self.correct_top_1, tf.int32))
            self.correct_num_top_5 = tf.reduce_sum(tf.cast(self.correct_top_5, tf.int32))


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        self.vgg16_npy_path = vgg16_npy_path

    def build(self, initializer, mode_name='vgg16_model', istraining=True):
        with tf.variable_scope(mode_name, initializer=initializer):
            """
            load variable from npy to build the VGG
    
            :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
            """

            start_time = time.time()
            print("build VGG16 model started")

            #shared input
            self.keep_prob = tf.placeholder("float")

            # var init
            with tf.variable_scope('vgg16_conv'):
                with tf.variable_scope('conv1_1') as scope:
                    self.conv1_1_W = self.get_conv_filter([3, 3, FRAME_PER_CLIP, 64])
                    self.conv1_1_b = self.get_bias([64])
                with tf.variable_scope('conv1_2') as scope:
                    self.conv1_2_W = self.get_conv_filter([3, 3, 64, 64])
                    self.conv1_2_b = self.get_bias([64])
                with tf.variable_scope('conv2_1') as scope:
                    self.conv2_1_W = self.get_conv_filter([3, 3, 64, 128])
                    self.conv2_1_b = self.get_bias([128])
                with tf.variable_scope('conv2_2') as scope:
                    self.conv2_2_W = self.get_conv_filter([3, 3, 128, 128])
                    self.conv2_2_b = self.get_bias([128])        
                with tf.variable_scope('conv3_1') as scope:
                    self.conv3_1_W = self.get_conv_filter([3, 3, 128, 256])
                    self.conv3_1_b = self.get_bias([256])
                with tf.variable_scope('conv3_2') as scope:
                    self.conv3_2_W = self.get_conv_filter([3, 3, 256, 256])
                    self.conv3_2_b = self.get_bias([256])
                with tf.variable_scope('conv3_3') as scope:
                    self.conv3_3_W = self.get_conv_filter([3, 3, 256, 256])
                    self.conv3_3_b = self.get_bias([256])
                with tf.variable_scope('conv4_1') as scope:
                    self.conv4_1_W = self.get_conv_filter([3, 3, 256, 512])
                    self.conv4_1_b = self.get_bias([512])
                with tf.variable_scope('conv4_2') as scope:
                    self.conv4_2_W = self.get_conv_filter([3, 3, 512, 512])
                    self.conv4_2_b = self.get_bias([512])
                with tf.variable_scope('conv4_3') as scope:
                    self.conv4_3_W = self.get_conv_filter([3, 3, 512, 512])
                    self.conv4_3_b = self.get_bias([512])
                with tf.variable_scope('conv5_1') as scope:
                    self.conv5_1_W = self.get_conv_filter([3, 3, 512, 512])
                    self.conv5_1_b = self.get_bias([512])
                with tf.variable_scope('conv5_2') as scope:
                    self.conv5_2_W = self.get_conv_filter([3, 3, 512, 512])
                    self.conv5_2_b = self.get_bias([512])
                with tf.variable_scope('conv5_3') as scope:
                    self.conv5_3_W = self.get_conv_filter([3, 3, 512, 512])
                    self.conv5_3_b = self.get_bias([512])
            
            # it is easy to forget
            conv_output_shape = 7 * 7 * 512

            with tf.variable_scope('vgg16_fc'):
                with tf.variable_scope('fc6') as scope:
                    self.fc6_W = self.get_fc_weight([conv_output_shape, 4096])
                    self.fc6_b = self.get_bias([4096])
                with tf.variable_scope('fc7') as scope:
                    self.fc7_W = self.get_fc_weight([4096, 4096])
                    self.fc7_b = self.get_bias([4096])
                with tf.variable_scope('fc8') as scope:
                    self.fc8_W = self.get_fc_weight([4096, class_num])
                    self.fc8_b = self.get_bias([class_num])

            self.conv_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=mode_name + '/' + 'vgg16_conv')
            self.fc_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=mode_name + '/' + 'vgg16_fc')

            # model init
            self.models = []
            for i in range(num_gpu):
                with tf.device('/gpu:%d' % i):
                    model = Single_GPU_MODEL(self)
                    tf.add_to_collection('jh_loss', model.loss)
                    tf.add_to_collection('jh_corr_num_1', model.correct_num_top_1)
                    tf.add_to_collection('jh_corr_num_5', model.correct_num_top_5)
                    tf.add_to_collection('jh_grad_conv', model.grad_conv)
                    tf.add_to_collection('jh_grad_fc', model.grad_fc)
                    self.models.append(model)

            # evaluation
            with tf.device('/gpu:0'):
                losses = tf.get_collection('jh_loss')
                corr_nums_1 = tf.get_collection('jh_corr_num_1')
                corr_nums_5 = tf.get_collection('jh_corr_num_5')
                with tf.name_scope('avg_evaluation'): 
                    self.loss = tf.add_n(losses)
                    self.correct_num_top_1 = tf.add_n(corr_nums_1)
                    self.correct_num_top_5 = tf.add_n(corr_nums_5)

            if not istraining:
                print(("build VGG16 model finished: %ds" % (time.time() - start_time)))
                return

            # avg grad
            with tf.device('/gpu:0'):
                grads_conv = tf.get_collection('jh_grad_conv')        
                grads_fc = tf.get_collection('jh_grad_fc')
                with tf.name_scope('avg_grad'):
                    avg_grads_conv = []
                    avg_grads_fc = []
                    for grads_per_var in zip(*grads_conv):
                        grads_per_var = [tf.expand_dims(grad,0) for grad in grads_per_var]
                        grads_per_var = tf.concat(grads_per_var, 0)
                        grad = tf.reduce_mean(grads_per_var, 0)
                        avg_grads_conv.append(grad)
                    for grads_per_var in zip(*grads_fc):
                        grads_per_var = [tf.expand_dims(grad,0) for grad in grads_per_var]
                        grads_per_var = tf.concat(grads_per_var, 0)
                        grad = tf.reduce_mean(grads_per_var, 0)
                        avg_grads_fc.append(grad)
                with tf.name_scope('optimize'):        
                    self.opt_conv = tf.train.MomentumOptimizer(0.000125, 0.9)
                    self.opt_fc = tf.train.MomentumOptimizer(0.000125, 0.9)
                    self.train_op_conv = self.opt_conv.apply_gradients(zip(avg_grads_conv, self.conv_var_list))
                    self.train_op_fc = self.opt_fc.apply_gradients(zip(avg_grads_fc, self.fc_var_list))
                    self.train_op = tf.group(self.train_op_conv, self.train_op_fc)

            print(("build VGG16 model finished: %ds" % (time.time() - start_time)))

    def load_pretrain_model(self, sess):
        if self.vgg16_npy_path is None:
            print('Warning: vgg16.npy file is not found! Can not load the pretrain model!')

            return
        else:
            self.data_dict = np.load(self.vgg16_npy_path, encoding='latin1')
            print("npy file loaded")

        print ('Warning: you have to initialize all variables before using this function!')

        # sess.run(self.conv1_1_W.assign(self.data_dict[()]['conv1_1f']))
        # sess.run(self.conv1_1_b.assign(self.data_dict[()]['conv1_1b']))
        sess.run(self.conv1_2_W.assign(self.data_dict[()]['conv1_2f']))
        sess.run(self.conv1_2_b.assign(self.data_dict[()]['conv1_2b']))

        sess.run(self.conv2_1_W.assign(self.data_dict[()]['conv2_1f']))
        sess.run(self.conv2_1_b.assign(self.data_dict[()]['conv2_1b']))
        sess.run(self.conv2_2_W.assign(self.data_dict[()]['conv2_2f']))
        sess.run(self.conv2_2_b.assign(self.data_dict[()]['conv2_2b']))

        sess.run(self.conv3_1_W.assign(self.data_dict[()]['conv3_1f']))
        sess.run(self.conv3_1_b.assign(self.data_dict[()]['conv3_1b']))
        sess.run(self.conv3_2_W.assign(self.data_dict[()]['conv3_2f']))
        sess.run(self.conv3_2_b.assign(self.data_dict[()]['conv3_2b']))
        sess.run(self.conv3_3_W.assign(self.data_dict[()]['conv3_3f']))
        sess.run(self.conv3_3_b.assign(self.data_dict[()]['conv3_3b']))

        sess.run(self.conv4_1_W.assign(self.data_dict[()]['conv4_1f']))
        sess.run(self.conv4_1_b.assign(self.data_dict[()]['conv4_1b']))
        sess.run(self.conv4_2_W.assign(self.data_dict[()]['conv4_2f']))
        sess.run(self.conv4_2_b.assign(self.data_dict[()]['conv4_2b']))
        sess.run(self.conv4_3_W.assign(self.data_dict[()]['conv4_3f']))
        sess.run(self.conv4_3_b.assign(self.data_dict[()]['conv4_3b']))

        sess.run(self.conv5_1_W.assign(self.data_dict[()]['conv5_1f']))
        sess.run(self.conv5_1_b.assign(self.data_dict[()]['conv5_1b']))
        sess.run(self.conv5_2_W.assign(self.data_dict[()]['conv5_2f']))
        sess.run(self.conv5_2_b.assign(self.data_dict[()]['conv5_2b']))
        sess.run(self.conv5_3_W.assign(self.data_dict[()]['conv5_3f']))
        sess.run(self.conv5_3_b.assign(self.data_dict[()]['conv5_3b']))

        sess.run(self.fc6_W.assign(self.data_dict[()]['fc6f']))
        sess.run(self.fc6_b.assign(self.data_dict[()]['fc6b']))
        sess.run(self.fc7_W.assign(self.data_dict[()]['fc7f']))
        sess.run(self.fc7_b.assign(self.data_dict[()]['fc7b']))
        self.data_dict = []
        print('The pre_train model have been successfully loaded!')

    def get_conv_filter(self, filter_size):
        return tf.get_variable('filter', filter_size, dtype=tf.float32)

    def get_bias(self, biases_size):
        return tf.get_variable('biases', biases_size, dtype=tf.float32)

    def get_fc_weight(self, weight_size):
        return tf.get_variable('weights', weight_size, dtype=tf.float32)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.allow_soft_placement = True #allow put calc to cpu when not support gpu  
    config.gpu_options.allow_growth = True
        
    with tf.Session(config=config) as sess:
        #vgg16 = Vgg16('./two_stream_vgg16.npy')
        vgg16 = Vgg16()
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        vgg16.build(initializer)
        sess.run(tf.global_variables_initializer())
        vgg16.load_pretrain_model(sess)
