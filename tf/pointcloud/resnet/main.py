from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio
import logging
import time
import os,sys
import traceback
import pdb
from numba import jit
from tensorflow.contrib.keras import backend as K
from multiprocessing import Pool
import cProfile

import resnet
import dataset
import utils

logger = utils.make_logger('main', is_stdout=False, filename='./log/main.log')
args = utils.args

def get_learn_rate(epoch):
    if epoch <= 24:
        return args.learn_rate
    elif epoch <= 32:
        return 1e-4
    else:
        return 3e-5

def run_epoch(epoch, model, sess, data, is_train=True):
    logger.info('begin epoch {}'.format(epoch))

    if is_train:
        train_op = model.train_op
    else:
        train_op = tf.no_op()
    learn_rate = get_learn_rate(epoch)

    data.reset()

    sum_acc = 0
    sum_loss = 0
    st_time = time.time()

    while not data.is_end():
        batch = data.get_batch()

        feed = {model.input_img:batch[0],model.input_label:batch[1]}
        feed[model.input_lr] = learn_rate
        feed[K.learning_phase()] = int(is_train)
        calc_obj = [train_op,model.acc_num,model.loss]
        calc_ans = sess.run(calc_obj, feed_dict=feed)

        sum_acc += calc_ans[1]
        sum_loss += calc_ans[2]

        if data.batch_cnt % 20 == 0:
            s = '[batch {}]acc:{:.3} loss:{:.3} avg-time:{:.5}'
            avg_acc = sum_acc/data.data_cnt
            avg_loss = sum_loss/data.batch_cnt
            avg_time = (time.time()-st_time)/data.batch_cnt
            s = s.format(data.batch_cnt,avg_acc,avg_loss,avg_time)
            logger.info(s)
            print(s)

    logger.info('[epoch {}]End..batch_num:{},data_num:{}'.format(epoch,data.batch_cnt,data.data_cnt))

    avg_acc = sum_acc / data.data_cnt
    avg_loss = sum_loss / data.batch_cnt

    return avg_acc,avg_loss

def run_test(epoch, model, sess, times, len_accs):
    logger.info('begin test epoch {}'.format(epoch))
    st_time = time.time()

    with open(args.testid_path) as f:###
        ids = f.readline().strip().split(',')

    #get test probe
    probe_features = []

    for id_ in ids:
        #path = os.path.join(args.probe_dir, id_.zfill(3), 'slice233-feature.mat')
        #if os.path.exists(path):
        if False:
            mat = sio.loadmat(path)
            features = mat['features']
        else:
            path = os.path.join(args.probe_dir, id_.zfill(3), 'slice0.1.mat')
            mat = sio.loadmat(path)
            imgs = mat['slice']

            imgs = dataset.preprocess_image(imgs)

            feed = {model.input_img:imgs}
            feed[K.learning_phase()] = 0
            calc_obj = [model.output_feature]
            calc_ans = sess.run(calc_obj, feed_dict=feed)

            features = calc_ans[0]

            #path = os.path.join(args.probe_dir, id_.zfill(3), 'slice233-100-feature.mat')
            #sio.savemat(path, {'features':features}, do_compression=True)

        probe_features.append(features)

    logger.info('get features of probe end')

    # get gallery feature
    ixs = []
    imgs = []
    for id_ in ids:
        path = os.path.join(args.gallery_dir, id_.zfill(3), 'slice0.1.mat')
        mat = sio.loadmat(path)
        len_ = len(mat['slice'])
        id_ixs = []
        id_imgs = []
        for i in range(times):
            ix = np.random.randint(len_)
            img = mat['slice'][ix]
            id_imgs.append(img)
            id_ixs.append(ix)
        ixs.append(id_ixs)
        imgs.append(id_imgs)

    ixs = list(zip(*ixs))
    imgs = list(zip(*imgs))

    gallery_features = []
    for i in range(times):
        img = imgs[i]
        img = np.array(img)

        img = dataset.preprocess_image(img)

        feed = {model.input_img:img}
        feed[K.learning_phase()] = 0
        calc_obj = [model.output_feature]
        calc_ans = sess.run(calc_obj, feed_dict=feed)

        features = calc_ans[0]
        gallery_features.append(features)

        #logger.info('{}'.format(str(ixs)))

    #calc acc
    sum_imgs = 0
    for id_features in probe_features:
        sum_imgs += len(id_features)
    sum_accs = np.zeros(len_accs)

    p_args = []
    for i in range(times):
        p_args.append([ids,probe_features,gallery_features[i],len_accs])

    pool = Pool(times)
    res = pool.map(proc_calc, p_args)
    pool.close()
    pool.join()

    for one_accs in res:
        #print(one_accs/sum_imgs)
        sum_accs += one_accs

    avg_accs = sum_accs / (sum_imgs*times)
    print(avg_accs)
    logger.info('[epoch {}]{}'.format(epoch,avg_accs))

    return avg_accs[0],int(time.time()-st_time)

def proc_calc(p_args):
    ids,probe_features,gallery_features,len_accs = p_args
    one_accs = np.zeros(len_accs)
    for j,id_ in enumerate(ids):
        id_features = probe_features[j]
        label_ix = j
        accs = calc_acc(gallery_features, id_features, label_ix, len_accs)
        one_accs += accs
    return one_accs

@jit
def calc_acc(features, batch_features, label_ix, len_accs):
    accs = np.zeros((len_accs), dtype=np.int32)

    for batch_feature in batch_features:
        dists = []

        for j,feature in enumerate(features):
            len1 = np.sqrt(np.sum(batch_feature**2))
            len2 = np.sqrt(np.sum(feature**2))
            dist = np.abs(np.sum(batch_feature*feature))/(len1*len2)
            dists.append(dist)

        arg_dists = np.argsort(dists)
        arg_ix = -1
        for j in range(len(arg_dists)):
            if arg_dists[j] == label_ix:
                arg_ix = j
                break

        top = len(arg_dists)-1-arg_ix
        if top < len_accs:
            for j in range(top,len_accs):
                accs[j] += 1

    return accs

def train():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = resnet.ResNet(args)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=100)
    model.load_model(sess, saver)
    logger.info('model loaded')

    train_config = {'data_dir':args.train_dir}
    dtrain = dataset.DataSet(train_config)
    logger.info('dataset loaded')

    for epoch in range(args.beg_epoch, args.epochs):
        train_acc,train_loss = run_epoch(epoch, model, sess, dtrain)

        saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'), global_step=epoch)
        logger.info('model of epoch {} saved'.format(epoch))

        s = '[epoch {}]train-acc:{:.3} train-loss:{:.3}'
        s = s.format(epoch,train_acc,train_loss)
        logger.warning(s)
        print(s)

        #if epoch!=args.beg_epoch and (epoch-args.beg_epoch)%10==0:
        if True:
            valid_acc,valid_time = run_test(epoch, model, sess, 10, 20)
            s = '[epoch {}]test-acc:{:.3} test-time:{}'.format(epoch, valid_acc, valid_time)
            logger.warning(s)
            print(s)

    logger.info('train end')

def test():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = resnet.ResNet(args)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=100)

    for epoch in range(0, args.beg_epoch):
        args.model_path = os.path.join(args.model_dir, 'model.ckpt-{}'.format(epoch))
        model.load_model(sess, saver)
        valid_acc,valid_time = run_test(epoch, model, sess, 10, 20)
        s = '[epoch {}]test-acc:{:.3} test-time:{}'.format(epoch, valid_acc, valid_time)
        logger.warning(s)
        print(s)
    return

    logger.info('test end')

def unittest():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = resnet.ResNet(args)
    sess.run(tf.global_variables_initializer())

def main():
    try:
        #cProfile.run('train()', 'ret.txt')
        train()
        #test()
        #unittest()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("{}\n{}\n{}\n".format(exc_type,exc_value,traceback.format_exc()))
    finally:
        dataset.close_threads()

if __name__ == '__main__':
    main()
