import numpy as np
import scipy
import scipy.io as sio
import os
import time
import pdb
from numba import jit

import utils
import subthread

logger = utils.make_logger('dataset', is_stdout=False, filename='./log/dataset.log')

class MyThread(subthread.subthread):
    def work(self, args):
        filepath = args
        mat = sio.loadmat(filepath)
        logger.info('load file:{}'.format(filepath))
        return mat

threads = []

def add_thread(thread):
    global threads
    threads.append(thread)

def close_threads():
    for thread in threads:
        thread.setExit()
    for thread in threads:
        thread.join()


def preprocess_image(images):
    board = np.zeros([images.shape[0],230,102,20], dtype=np.uint8)
    begH = 3
    endH = begH + utils.args.img_height
    begW = 1
    endW = begW + utils.args.img_width
    begC = 1
    endC = begC + utils.args.img_channels
    board[:,begH:endH,begW:endW,begC:endC] = images

    board = np.expand_dims(board, -1)
    #board = np.divide(board, 255.0)

    return board


class DataSet(object):
    def __init__(self, config, is_train=True):
        self.is_train = is_train

        self.data_dir = config['data_dir']
        self.data = None
        self.data_cnt = 0
        self.data_ibeg = 0

        self.batch_size = utils.args.batch_size
        self.batch_cnt = 0

        self.filelist = os.listdir(self.data_dir)
        if is_train:
            self.filelist = ['slice233-part-{}.mat'.format(i) for i in range(0,100,5)]
            self.filelist += ['slice233-part-{}.mat'.format(i) for i in range(1,100,5)]
        else:
           self.filelist = ['slice233-part-{}.mat'.format(i) for i in range(25,50,5)]
        self.filelist = [os.path.join(self.data_dir,file) for file in self.filelist]
        self.file_num = len(self.filelist)
        self.file_cnt = self.file_num

        self.thread = MyThread()
        add_thread(self.thread)
        self.thread.start()

        logger.info('data_dir:{}'.format(self.data_dir))
        logger.info('file_num:{}'.format(self.file_num))
        logger.info('dataset inited!')

    def reset(self):
        self.data_cnt = 0
        self.batch_cnt = 0
        self.data_ibeg = 0
        self.file_cnt = 0
        self.data = None

        np.random.shuffle(self.filelist)

        logger.info('dataset reset')

        self.load_file()

    def get_batch(self):
        if self.is_end():
            logger.error('no batch to get any more')
            raise Exception('no batch to get any more')

        if self.data_ibeg == len(self.data[0]):
            self.load_file()

        #get batch
        ibeg = self.data_ibeg
        iend = min(self.data_ibeg + self.batch_size, len(self.data[0]))
        images = self.data[0][ibeg:iend]
        labels = self.data[1][ibeg:iend]

        #preprocess
        images = preprocess_image(images)

        #update status
        self.data_ibeg = iend
        self.batch_cnt += 1
        self.data_cnt += iend-ibeg

        return images,labels

    def is_end(self):
        return (self.data is None or self.data_ibeg == len(self.data[0])) and self.file_cnt == self.file_num

    def load_file(self):
        if self.file_cnt == self.file_num:
            logger.error('no file to read any more')
            raise Exception('no file to read any more')

        #first read
        if self.file_cnt == 0:
            self.thread.setOn(self.filelist[self.file_cnt])

        #get from thread
        while not self.thread.isOff():
            time.sleep(0.001)
        mat = self.thread.getOff()
        self.file_cnt += 1

        #set next read
        if self.file_cnt < self.file_num:
            self.thread.setOn(self.filelist[self.file_cnt])

        images = mat['images'] #(n,224,100,1)
        labels = mat['labels'][0] #(1,n)

        self.data = [images,labels]
        self.data_ibeg = 0
