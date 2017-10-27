import scipy.io as sio
import h5py
import numpy as np

#jh:17.10.16

#建议先用交互命令行一步一步执行
#mat文件除了第一步,后面没有统一的格式

def readmat(file):
    try:
        # version < v-7.3
        mat = sio.loadmat(file)
        print(mat.keys())
        data1 = mat['data1']
        data2 = mat['data2']
    except NotImplementedError as e:
        # version >= v-7.3
        print("warning:{} can't open by scipy.io.loadmat".format(file))
        mat = h5py.File(file)
        print(list(mat.keys()))
        data1 = mat['data1']
        data1 = np.array([np.transpose(mat[frame[0]]) for frame in data1])
        data2 = mat['data2']
        data2 = np.array([np.transpose(mat[frame[0]]) for frame in data2])
        # frame[0] is just HDF5 obj reference, and mat[frame[0]] get the data of this reference

    return data1,data2

def savemat(file, data1, data2):
    sio.savemat(file, {'data1':data1,'data2':data2}, do_compression=True)
    # do_compression可以压缩空间
