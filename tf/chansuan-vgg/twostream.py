from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

'''
jh:2017.9.4
multi gpu and mulit thread reading

note:
1. change version of vgg16 and dataset in import 
'''

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import traceback

from six.moves import xrange  # pylint: disable=redefined-builtin

import vgg16_8f as vgg16
import dataset5 as dataset

# Basic model parameters
FLAGS = None
# TRAIN_SIZE = 35875
# VALID_SIZE = 5784
# TEST_SIZE = 5784
TRAIN_SIZE = 63725
VALID_SIZE = 15933
TEST_SIZE = 15933
summary_global_step = [0, 0, 0]
short_term_loss = [6.9]
short_term_len = 100

ffout = open('knf.txt', 'w+')

class ModelConfig(object):
    init_scale = 0.05
    learning_rate = 0.01
    max_grad_norm = 5
    keep_epoch = 100
    max_epoch = 100
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 36
    num_class = 501


def dense_to_one_hot(labels_dense, num_class):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_class
    labels_one_hot = np.zeros((num_labels, num_class), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1.0
    return labels_one_hot


def load_model(sess, saver, ckpt_path, train_model):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)

    if latest_ckpt:
        print('resume from', latest_ckpt)

        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print('building model from scratch')
        sess.run(tf.global_variables_initializer())
        train_model.load_pretrain_model(sess)
        return -1


def run_epoch(session, keep_prob, log_file, batch_size, model, data, eval_op, step_index, summary_writer,
              istraining=True):
    """Runs the model on the given data."""

    data_size = data.data_size

    #dataset has made sure data_size % (batch_size*num_gpu) == 0
    iterations = data_size // batch_size // vgg16.num_gpu

    data.reset()

    start_time = time.time()

    all_correct_top_1 = 0
    all_correct_top_5 = 0
    total_loss = 0.0

    t_b_time = 0.0##
    t_t_time = 0.0##

    for iter in range(iterations):
        fetches = [model.correct_num_top_1, model.correct_num_top_5, model.loss, eval_op]

        feed_dict = {}
        for i in range(vgg16.num_gpu):
            sub_model = model.models[i]

            s_b_time = time.time()##

            data_features, data_labels = data.next_batch()

            s_e_time = time.time()##
            t_b_time += s_e_time - s_b_time##

            data_labes_one_hot = dense_to_one_hot(data_labels, 501)

            feed_dict[sub_model.labels] = data_labels
            feed_dict[sub_model.flow] = data_features
            feed_dict[sub_model.labels_one_hot] = data_labes_one_hot

        feed_dict[model.keep_prob] = keep_prob

        s_b_time = time.time()##

        correct_top_1, correct_top_5, loss, _ = session.run(fetches, feed_dict)##

        s_e_time = time.time()##
        t_t_time += s_e_time - s_b_time##

        assert not np.isnan(loss), 'loss = NaN'##
        total_loss += loss##

        all_correct_top_1 += correct_top_1##
        all_correct_top_5 += correct_top_5##

        if istraining:
            short_term_loss.insert(0, loss)##
            if len(short_term_loss) > short_term_len:
                short_term_loss.pop()
            cur_short_term_loss = sum(short_term_loss) * 1.0 / (len(short_term_loss)*vgg16.num_gpu)

            if iter % 5 == 1:
                info = "%.3f: acc(top 1): %.3f acc(top 5): %.3f loss %.3f speed: %.3f sec" % (
                    iter * 1.0 / iterations, 
                    correct_top_1 * 1.0 / (batch_size*vgg16.num_gpu), ##
                    correct_top_5 * 1.0 / (batch_size*vgg16.num_gpu), ##
                    cur_short_term_loss,
                    (time.time() - start_time) * 1.0 / ((iter + 1) * (batch_size*vgg16.num_gpu))
                )

                print(info)
                log_file.write(info + '\n')
                a_b_time = t_b_time/((iter + 1)*vgg16.num_gpu*batch_size)
                a_t_time = t_t_time/((iter + 1)*vgg16.num_gpu*batch_size)
                log_file.write("avg-io:{} avg-run:{}\n".format(a_b_time,a_t_time))
                log_file.flush()

        else:
            if iter % 5 == 1:
                info = "%.3f: acc(top 1): %.3f acc(top 5): %.3f loss %.3lf speed: %.3f sec" % (
                    iter * 1.0 / iterations, 
                    correct_top_1 * 1.0 / (batch_size*vgg16.num_gpu), 
                    correct_top_5 * 1.0 / (batch_size*vgg16.num_gpu),
                    total_loss * 1.0 / ((iter + 1)*vgg16.num_gpu),
                    (time.time() - start_time) * 1.0 / ((iter + 1) * (batch_size*vgg16.num_gpu))
                )
                print(info)
                log_file.write(info + '\n')
                a_b_time = t_b_time/((iter + 1)*vgg16.num_gpu*batch_size)
                a_t_time = t_t_time/((iter + 1)*vgg16.num_gpu*batch_size)
                log_file.write("avg-io:{} avg-run:{}\n".format(a_b_time,a_t_time))
                log_file.flush()

    if istraining:
        return all_correct_top_1 * 1.0 / (iterations * batch_size*vgg16.num_gpu), all_correct_top_5 * 1.0 / (
            iterations * batch_size*vgg16.num_gpu), total_loss * 1.0 / (iterations*vgg16.num_gpu)
    else:
        return all_correct_top_1 * 1.0 / data_size, all_correct_top_5 * 1.0 / data_size, total_loss * 1.0 / iterations


def run_train():
    fout = open('inf.txt', 'w+')
    train_config = ModelConfig()

    eval_config = ModelConfig()
    eval_config.keep_prob = 1.0

    test_config = ModelConfig()
    test_config.keep_prob = 1.0

    Session_config = tf.ConfigProto(allow_soft_placement=True)
    Session_config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=Session_config) as sess:
        # if True:
        initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                    train_config.init_scale)

        train_model = vgg16.Vgg16(FLAGS.vgg16_file_path)
        train_model.build(initializer)

        data_train = dataset.DataSet(FLAGS.file_path_train, TRAIN_SIZE, train_config.batch_size)
        data_valid = dataset.DataSet(FLAGS.file_path_valid, VALID_SIZE, eval_config.batch_size, is_train_set=False)
        data_test = dataset.DataSet(FLAGS.file_path_test, TEST_SIZE, test_config.batch_size, is_train_set=False)

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        saver = tf.train.Saver(max_to_keep=100)
        last_epoch = load_model(sess, saver, FLAGS.saveModelPath, train_model)
        print('start: ', last_epoch + 1)

        for i in range(last_epoch + 1, train_config.max_epoch):
            train_accury_1, train_accury_5, train_loss = run_epoch(sess, train_config.keep_prob, fout,
                                                                   train_config.batch_size, train_model, data_train,
                                                                   train_model.train_op, 0, train_writer)

            info = "Epoch: %d Train acc(top 1): %.3f Train acc(top 5): %.3f Loss: %.3f" % (
                i, train_accury_1, train_accury_5, train_loss)
            print(info)
            fout.write(info + '\n')
            fout.flush()

            saver.save(sess, FLAGS.saveModelPath + '/model.ckpt', global_step=i)

            if i != 0 and i % 10 == 0:
                valid_accury_1, valid_accury_5, valid_loss = run_epoch(sess, eval_config.keep_prob, fout,
                                                                       eval_config.batch_size, train_model, data_valid,
                                                                       tf.no_op(), 1, valid_writer, istraining=False)

                info = "Epoch: %d Valid acc(top 1): %.3f Valid acc(top 5): %.3f Loss: %.3f" % (
                    i, valid_accury_1, valid_accury_5, valid_loss)
                print(info)
                fout.write(info + '\n')
                fout.flush()

        test_accury_1, test_accury_5, test_loss = run_epoch(sess, test_config.keep_prob, fout,
                                                            test_config.batch_size, train_model, data_test,
                                                            tf.no_op(), 2, test_writer, istraining=False)

        info = "Final: Test acc(top 1): %.3f Test acc(top 5): %.3f Loss %.3f" % (
            test_accury_1, test_accury_5, test_loss)
        print(info)
        fout.write(info + '\n')
        fout.flush()

        train_writer.close()
        valid_writer.close()
        test_writer.close()
        
        # close multi thread
        data_train.close()
        data_valid.close()
        data_test.close()
        dataset.pool.close()

        print("Training step is compeleted!")
        fout.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
       tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.saveModelPath):
       tf.gfile.MakeDirs(FLAGS.saveModelPath)

    try:
        run_train()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        ffout.write("{}\n{}\n{}\n".format(exc_type,exc_value,traceback.format_exc()))
        ffout.flush()
        dataset.pool.close()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vgg16_file_path',
        type=str,
        default='./two_stream_vgg16.npy',
        # default=None,
        help='file path of vgg16 pretrain model.'
    )
    parser.add_argument(
        '--file_path_train',
        type=str,
        default='../labels/train_label_len',
        help='file_path is the path of [video_path label] file.'
    )
    parser.add_argument(
        '--file_path_valid',
        type=str,
        default='../labels/valid_label_len',
        help='file_path is the path of [video_path label] file.'
    )
    parser.add_argument(
        '--file_path_test',
        type=str,
        default='../labels/valid_label_len',
        help='file_path is the path of [video_path label] file.'
    )
    parser.add_argument(
        '--data_root_dir',
        type=str,
        default='/home/share2/chaLearn-Iso/Seq/',
        help='data_root_dir is the root used for video_path,so we can use data_root_dir +  video_path to access video.'
    )
    parser.add_argument(
        '--saveModelPath',
        type=str,
        default='../ModelPara',
        help='Directory to put model parameter.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../Modellog',
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
