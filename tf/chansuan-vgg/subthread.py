from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import traceback
import numpy as np
import random

import threading
import time

'''
jh:2017.9.3
redident thread
'''

class subthread(threading.Thread):
    '''
    Quick-lock, shared-memory, resident thread

    on:args
    off:ret
    1. main thread set `off` = none and set `on` = something
    2. sub thread find `on` != none, and begin work
    3. main thread find `off` is still none, so sub thread is working
    4. sub thread finish and set `on` = none and set `off` = something
    5. main thread find `off` != none, so sub thread finish working
    6. sub thread find `on` is none, and sleep... 
    '''
    def __init__(self, args={}):
        '''
        args: dict, as the shared memory
        '''
        super(subthread, self).__init__()
        self.args = args if len(args) != 0 else {}
        self.args['end'] = False
        self.args['on'] = None
        self.args['off'] = True
        self.sleeptime = 0.001

    def setExit(self):
        '''
        ask subthread to exit
        '''
        self.args['end'] = True

    def setOn(self, s):
        '''
        give args and ask subthread to work
        s: obj, args of the work 
        '''
        assert self.isOff(), 'thread is working'
        self.args['off'] = None
        self.args['on'] = s

    def getOff(self):
        '''
        get the return value
        ret: obj, get the return value of the work
        '''
        assert self.isOff(), 'thread is working'
        return self.args['off']

    def isOff(self):
        '''
        ret: bool, return if work finished
        '''
        return self.args['off'] is not None

    def run(self):
        while True:
            # judge end
            if self.args['end']:
                return
            # judge sleep
            if self.args['on'] is None:
                time.sleep(self.sleeptime)
                continue
            # work
            args = self.args['on']
            try:
                ret = self.work(args)
            except Exception as e:
                ret = self.solve_error(e)
            # finish
            self.args['on'] = None
            self.args['off'] = ret

    def work(self, args):
        '''
        virtual func: your work
        '''
        print("please rewrite this func.")

    def solve_error(e):
        '''
        virtual func: how to solve error
        '''
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("{}\n{}\n{}\n".format(exc_type,exc_value,traceback.format_exc()))
        raise


#your code
import skimage
import skimage.io
from matplotlib import pyplot as plt

class MyThread(subthread):
    def work(self, args):
        filepath = args
        if not isinstance(filepath, str):
                filepath = filepath.decode()
        img = skimage.io.imread(filepath)
        return img

    def solve_error(e):
        return None


if __name__ == '__main__':
    st = MyThread()
    st.start()

    st.setOn(r"D:\jianheng\a.jpg")
    while not st.isOff():
        time.sleep(0.1)
    img = st.getOff()    

    if img is None:
        print("Path not valid")
    else:
        skimage.io.imshow(img)
        plt.show()

    st.setExit()
    st.join()
