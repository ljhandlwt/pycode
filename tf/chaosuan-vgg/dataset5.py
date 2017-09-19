from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
jh:2017.9.4
multi thread

implement:
use middle thread imply multi thread to read videos
use sub thread imply multi thread to read freams of a video 

here define 4 middle threads, and FRAME_PER_CLIP = 8
so total 4*8=32 sub threads

detail implements:
x1. main thread make several middle threads and start them(`DataSet.__init__`)
x2. main thread make filepath-queue(`DataSet.reset`)

1. middle thread get a filelist from the queue(`MiddleThread.work`)
2. middle thread decide which frames it will read(`MiddleThread.make_filelist`)
3. middle thread call `load_all_images`(`MiddleThread.work`)
4. `load_all_images` send frame-filepath to subthread(`load_all_images`)
5. subthread get the filepath, read it and return(`subthread.work`)
6. `load_all_images` get all frames and return(`load_all_images`)
7. middle thread preprocess the frames, and push it to batch-list(`MiddleThread.work`)

x3. main thread get a batch from batch-list(`DataSet.next_batch`)

genernal usage of DataSet:
```
import dataset
data = dataset.DataSet(...)
for i in range(epochs):
    data.reset()
    for j in range(batch_num):
        batch = data.next_batch()
        ...
data.close()
dataset.pool.close()        
```

note:
1. it add additional record to make sure tot_size % (batch_size*num_gpu) == 0
   so if run test, must dereplication after running model
2. before the proc close, don't forget to close the threads by `pool.close()`
3. don't let a dataset reset to read, when another dataset is reading
   it will crash because all dataset share the same pool(subthreads)
'''

import os
import numpy as np
import random

import skimage
import skimage.io
import skimage.transform
import skimage.color

import pdb
import sys
import time

from collections import deque
import threading
import traceback

import subthread

HEIGHTSIZE = 240
WIDTHSIZE = 320

HEIGHTRESIZE = 256
WIDTHRESIZE = 256

# please keep the same with the para in vgg16.py file
FRAME_PER_CLIP = 8
CROPSIZE = 224
num_gpu = 4

num_thread = 4 #here is middle-threads

fout = open('jnf.txt', 'w+')

# subthread pool
# genernal not need to change it
class Pool(object):
    def __init__(self, num_tfthread, num_thread):
        self.num_tfthread = num_tfthread
        self.num_thread = num_thread
        self.threads = []
        for i in range(self.num_tfthread):
            gpu_threads = []
            for j in range(self.num_thread):
                thread = subthread.subthread()
                thread.setDaemon(True)
                thread.start()
                gpu_threads.append(thread)
            self.threads.append(gpu_threads)    

    def set_work(self, tfthread_id, thread_id, file_path):
        thread = self.threads[tfthread_id][thread_id]
        assert thread.isOff(), '{}-{} is working!!!'.format(tfthread_id,thread_id)
        thread.setOn(file_path)
        return

    def get_ans(self, tfthread_id, thread_id):
        thread = self.threads[tfthread_id][thread_id]
        if thread.isOff():
            return thread.getOff()
        else:
            return None

    def close(self):
        for i in range(self.num_tfthread):
            for j in range(self.num_thread):
                self.threads[i][j].setExit()
        return

pool = Pool(num_thread, FRAME_PER_CLIP) #only one instance

# as thread in tf.FIFOQueue
class MiddleThread(threading.Thread):
    def __init__(self, args):
        super(MiddleThread, self).__init__()
        '''
        args:dict,it should include:
            id: int, id of this thread
            queue: deque, queue of filepath
            blist: list, list of batches(shape:[num_batches, batch_size], base is [8frame-img,label])
            q_lock: lock, lock of queue
            b_lock: lock, lock of batch_lists
            batch_size: int, size of batch
        '''

        self.args = args
        self.args['end'] = False
        # if no task, files of a epoch have been read
        # so don't need to wake up too frequently
        self.sleeptime = 1 

    def setExit(self):
        self.args['end'] = True

    def run(self):
        try:
            self.work()
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            fout.write("{}\n{}\n{}\n".format(exc_type,exc_value,traceback.format_exc()))
            fout.flush()
            raise    

    def work(self):    
        while True:
            # end judge
            if self.args['end']:
                return
            # task judge    
            self.args['q_lock'].acquire()
            if len(self.args['queue']) == 0:
                data = None
            else:
                data = self.args['queue'].pop()
            self.args['q_lock'].release()

            if data is None: # sleep
                time.sleep(self.sleeptime)
                continue

            # work
            file_list = self.make_filelist(data[0], data[1], data[2])
            imgs = load_all_images(file_list, self.args['id'])

            # preprocessing
            imgs = list(map(skimage.color.rgb2gray, imgs))
            imgs = np.stack(imgs, 2)
            x_beg = random.randint(0,HEIGHTRESIZE-CROPSIZE)
            y_beg = random.randint(0,WIDTHRESIZE-CROPSIZE)
            imgs = imgs[x_beg:x_beg+CROPSIZE,y_beg:y_beg+CROPSIZE,:]

            # push
            img = imgs
            label = data[1]
            ret = [img, label]
            self.args['b_lock'].acquire()
            blist = self.args['blist']
            if len(blist) == 0 or len(blist[-1]) == self.args['batch_size']:
                blist.append([ret])
            else:
                blist[-1].append(ret) 
            self.args['b_lock'].release()

    def make_filelist(self, path, label, len_):
        # decide which frames selected
        per_len = len_ / FRAME_PER_CLIP
        random_ixs = np.random.randint(0, per_len, size=(FRAME_PER_CLIP))
        base_ixs = np.arange(FRAME_PER_CLIP) * per_len
        ixs = (base_ixs+random_ixs).astype(np.int32) + 1 # 1~len, not 0~len-1
        ixs = np.maximum(ixs, 1)
        ixs = np.minimum(ixs, len_)
        # make file list
        video_name = path.split('/')[-1]
        base_dir = path + '/' + video_name + '-'
        file_list = [base_dir + str(ix) + '.jpg' for ix in ixs]
        return file_list


def load_all_images(file_list, tfthread_id):
    '''load add imgs of the same video'''

    for i in range(FRAME_PER_CLIP):
        pool.set_work(tfthread_id, i, file_list[i])
    res = []
    waitlist = list(range(FRAME_PER_CLIP))

    while len(waitlist) > 0:
        cur_len = len(waitlist)
        cnt_len = 0
        while cnt_len < cur_len:
            ans = pool.get_ans(tfthread_id, waitlist[cnt_len])
            if ans is not None:
                res.append(ans)
                waitlist.pop(cnt_len)
                break
            cnt_len += 1
        if cnt_len == cur_len:
            time.sleep(0.001)

    assert len(res) == FRAME_PER_CLIP

    return res


class DataSet(object):
    def __init__(self, file_path, data_size, batch_size, is_train_set=True):
        self.file_path = file_path
        self.data_size = data_size
        self.batch_size = batch_size
        self.is_train = is_train_set

        iters = self.data_size // self.batch_size // num_gpu
        if iters * self.batch_size * num_gpu < data_size:
            iters += 1
        self.data_size = iters * self.batch_size * num_gpu
        self.add_num = self.data_size - data_size # additional record num
        self.batch_num = self.data_size // self.batch_size
        self.batch_cnt = self.batch_num # important var in dataset   

        fout.write("{} {} {} {}\n".format(self.data_size, self.batch_num, self.add_num, self.batch_cnt))
        fout.flush()  

        # make path-list,label,len
        self.feature_path = []
        self.feature_label = []
        self.feature_len = []
        with open(file_path) as f:
            for i,line in enumerate(f):
                if i == data_size: #note: old data_size
                    break
                path, label, len_ = line.strip().split()
                self.feature_path.append(path)
                self.feature_label.append(int(label))
                self.feature_len.append(int(len_))
        assert len(self.feature_path) == self.data_size - self.add_num
        assert len(self.feature_path) >= self.add_num        
        if self.add_num > 0:
            self.feature_path.extend(self.feature_path[:self.add_num]) # add additional record          
            self.feature_label.extend(self.feature_label[:self.add_num])
            self.feature_len.extend(self.feature_len[:self.add_num])
        self.feature_path = np.array(self.feature_path)
        self.feature_label = np.array(self.feature_label)
        self.feature_len = np.array(self.feature_len)

        # make multi thread
        self.queue = deque()
        self.blist = deque()
        self.q_lock = threading.Lock()
        self.b_lock = threading.Lock()

        self.threads = []
        for i in range(num_thread):
            args = {
                'id':i,
                'queue':self.queue,
                'blist':self.blist,
                'q_lock':self.q_lock,
                'b_lock':self.b_lock,
                'batch_size':self.batch_size
            }
            thread = MiddleThread(args)
            thread.setDaemon(True)
            thread.start()
            self.threads.append(thread)

    def reset(self):
        '''
        clear rest batch and read batch from beginning
        generally, run a epoch and reset one time
        note: don't forget reset before the first epoch
        '''
        # clear rest batch
        while self.batch_cnt < self.batch_num:
            self.next_batch()
        self.batch_cnt = 0

        # make random
        data_ixs = np.arange(self.data_size)
        np.random.shuffle(data_ixs)

        feature_path = self.feature_path[data_ixs] 
        feature_label = self.feature_label[data_ixs]
        feature_len = self.feature_len[data_ixs]

        assert len(self.queue) == 0
        assert len(self.blist) == 0

        # en queue
        self.queue.extend(zip(self.feature_path, self.feature_label, self.feature_len))

    def next_batch(self):
        '''
        get a batch
        if no batch left, it will throw exception
        '''
        assert self.batch_cnt < self.batch_num, 'not rest batch'
        while len(self.blist) == 0 or len(self.blist[0]) < self.batch_size:
            time.sleep(0.001)
        self.b_lock.acquire()
        batch = self.blist.popleft()
        self.b_lock.release()    
        input_feature,input_label = list(zip(*batch))
        input_feature = np.array(input_feature, dtype=np.float32)
        input_label = np.array(input_label, dtype=np.int32)
        self.batch_cnt += 1

        return input_feature,input_label

    def close(self):
        '''
        close the reading
        after calling the func, you can't get batch any more
        '''
        for i in range(num_thread):
            self.threads[i].setExit()    

    def get_frame_nums(self, len_):
        '''
        make random frame nums of a video
        '''
        per_len = tf.cast(len_ / FRAME_PER_CLIP, tf.float32) # float64->float32
        random_ixs = tf.random_uniform([FRAME_PER_CLIP], 0, per_len)
        base_ixs = tf.range(0, FRAME_PER_CLIP, dtype=tf.float32) * per_len
        ixs = tf.cast(base_ixs + random_ixs, tf.int32) + 1 # 1~len, not 0~len-1
        ixs = tf.clip_by_value(ixs, 1, len_)
        return ixs
        

if __name__ == '__main__':
    ds = DataSet('videolist.txt', 5, 2)

    ds.reset()
    pdb.set_trace()

    ds.close()
    pool.close()
    print("ok")
