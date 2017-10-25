#jh:17.10.25
#after resnet3d for pointcloud

#use keras
#multi-gpu for keras
#model-dataset-run mode

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import pdb
import tensorflow.contrib.keras.api.keras.layers
from tensorflow.contrib.keras.api.keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Flatten,Activation
from tensorflow.contrib.keras import backend as K
import argparse
import time

args = None

class SubCNN(object):
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id

        with tf.device('/gpu:{}'.format(self.gpu_id)):
            self.init_input()
            self.init_model()
            self.init_loss()

    def init_input(self):
        self.input_img = tf.placeholder(tf.float32, [None,28*28])
        self.input_label = tf.placeholder(tf.int32, [None])
        self.input_droprate = tf.placeholder(tf.float32)

    def init_model(self):
        with tf.variable_scope('cnn', reuse=(self.gpu_id!=0)):
            x = self.input_img
            x = tf.reshape(x, (-1,28,28,1))
            x = Conv2D(20, (5,5), padding='same', activation='relu', name='conv1')(x)
            x = MaxPooling2D()(x)
            x = Conv2D(50, (5,5), padding='same', activation='relu', name='conv2')(x)
            x = MaxPooling2D()(x)
            x = Dropout(self.input_droprate)(x)
            x = Flatten()(x)
            x = Dense(500, activation='relu', name='fc1')(x)
            x = Dropout(self.input_droprate)(x)
            x = Dense(args.classes, activation='softmax', name='fc2')(x)

            self.output = x

    def init_loss(self):
        label_onehot = tf.one_hot(self.input_label, args.classes)
        cross_entropy = -tf.reduce_sum(label_onehot*tf.log(self.output+args.eps), axis=1)

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regular_vars = [var for var in train_vars if var.name.find('kernel')!=-1]
        regularizers = tf.add_n([tf.nn.l2_loss(var) for var in regular_vars])

        self.loss = tf.reduce_mean(cross_entropy) + args.weight_decay * regularizers
        self.grad = tf.gradients(self.loss, train_vars)

        corr_pred = tf.equal(tf.cast(self.input_label, tf.int64), tf.argmax(self.output, axis=1))
        self.acc_num = tf.reduce_sum(tf.cast(corr_pred, tf.int32))

        tf.add_to_collection('jh_loss', self.loss)
        tf.add_to_collection('jh_grad', self.grad)
        tf.add_to_collection('jh_acc', self.acc_num)

class CNN(object):
    def __init__(self):
        self.init_input()
        self.init_model()
        self.init_loss()

    def init_input(self):
        self.input_learnrate = tf.placeholder(tf.float32)

    def init_model(self):
        # make model
        self.models = []
        for i in range(args.num_gpu):
            submodel = SubCNN(i)
            self.models.append(submodel)

    def init_loss(self):
        with tf.device('/gpu:0'):
            # loss
            losses = tf.get_collection('jh_loss')
            loss_sum = tf.add_n(losses)
            self.loss = loss_sum / args.num_gpu
            # acc
            accs = tf.get_collection('jh_acc')
            self.acc_num = tf.add_n(accs)
            # avg-grad and train
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads = tf.get_collection('jh_grad')
            avg_grads = []
            for grads_per_var in zip(*grads):
                grads_per_var = tf.stack(grads_per_var, axis=0)
                grad = tf.reduce_mean(grads_per_var, 0)
                avg_grads.append(grad)
            self.opt = tf.train.AdamOptimizer(self.input_learnrate)
            self.train_op = self.opt.apply_gradients(zip(avg_grads, train_vars))

    def get_feed(self, batches, is_train, drop_rate, learn_rate=None):
        feed = {}
        for i in range(args.num_gpu):
            submodel = self.models[i]
            batch = batches[i]
            feed[submodel.input_img] = batch[0]
            feed[submodel.input_label] = batch[1]
            feed[submodel.input_droprate] = drop_rate
        feed[K.learning_phase()] = int(is_train)
        if learn_rate is not None:
            feed[self.input_learnrate] = learn_rate
        return feed


class DataSet(object):
    def __init__(self, data):
        self.data = data

        self.data_num = self.data[0].shape[0]
        self.batch_num = int(np.ceil(self.data_num/args.batch_size))

        self.data_ixs = None
        self.data_cnt = self.data_num
        self.batch_cnt = self.batch_num

    def reset(self):
        self.data_ixs = np.arange(self.data_num)
        np.random.shuffle(self.data_ixs)

        self.data_cnt = 0
        self.batch_cnt = 0

    def get_batch(self):
        assert not self.is_end()

        beg = self.data_cnt
        end = min(beg+args.batch_size, self.data_num)
        ixs = self.data_ixs[beg:end]

        images = self.data[0][ixs]
        labels = self.data[1][ixs]

        self.data_cnt = end
        self.batch_cnt += 1

        return images,labels

    def is_end(self):
        assert (self.data_cnt==self.data_num) == (self.batch_cnt==self.batch_num)
        return self.data_cnt==self.data_num


def run_epoch(epoch, model, sess, data, is_train=True):
    if is_train:
        train_op = model.train_op
        drop_rate = args.drop_rate
    else:
        train_op = tf.no_op()
        drop_rate = 0.0

    data.reset()

    sum_acc = 0
    sum_loss = 0.0
    st_time = time.time()

    while not data.is_end():
        batches = []
        for i in range(args.num_gpu):
            batches.append(data.get_batch())
            if data.is_end():
                break
        while len(batches) < 4:
            batches.append(batches[-1])

        feed = model.get_feed(batches, is_train, drop_rate, learn_rate=args.learn_rate)
        calc_obj = [train_op, model.acc_num, model.loss]
        calc_ans = sess.run(calc_obj, feed_dict=feed)

        sum_acc += calc_ans[1]
        sum_loss += calc_ans[2]

        #if data.batch_cnt % 10 == 0:
        #if is_train:
        if False:
            s = '[batch {}]train-acc:{:.3} train-loss:{:.3}'
            avg_acc = sum_acc / data.data_cnt
            avg_loss = sum_loss / data.batch_cnt
            print(s.format(data.batch_cnt,avg_acc,avg_loss))

    print(time.time()-st_time)

    avg_acc = sum_acc / data.data_num
    avg_loss = sum_loss / data.batch_num

    return avg_acc,avg_loss


def train():
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = CNN()
    sess.run(tf.global_variables_initializer())

    mnist = input_data.read_data_sets("MNIST_DATA/")
    dtrain = DataSet([mnist.train.images,mnist.train.labels])
    dtest = DataSet([mnist.test.images,mnist.test.labels])

    for epoch in range(args.epochs):
        train_acc,train_loss = run_epoch(epoch, model, sess, dtrain, is_train=True)
        test_acc,test_loss = run_epoch(epoch, model, sess, dtest, is_train=False)

        s = '[epoch {}]train-acc:{:.3} train-loss:{:.3}  test-acc:{:.3} test-loss:{:.3}'
        print(s.format(epoch,train_acc,train_loss,test_acc,test_loss))


def main():
    train()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_rate', default=1e-3, type=float)
    parser.add_argument('--drop_rate', default=0.5, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--num_gpu', default=2, type=int)
    parser.add_argument('--classes', default=10, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=100, type=int)

    global args
    args = parser.parse_args()


if __name__ == '__main__':
    make_args()
    main()