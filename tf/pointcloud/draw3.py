from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import scipy.io as sio
import argparse
import os,sys
import traceback
from numba import jit
import pdb
import time
import cProfile
import subthread
import h5py

#const
dist_threshold = 0.1
channels = 18
img_width = 100
img_height = 224

logfile = sys.stdout

class SaveThread(subthread.subthread):
    def work(self, args):
        save_path = args['save_path']
        outputs = args['outputs']
        sio.savemat(save_path, {'slice':outputs}, do_compression=True)
        print("save {}".format(save_path), file=logfile, flush=True)
        return 0

saveThread = SaveThread()

@jit
def rotate_z(poses, rad):
    '''
    :poses:array,shape(...,3)
    :rad:float
    :ret:array,shape(...,3)
    '''
    x = poses[...,0]
    y = poses[...,1]
    z = poses[...,2]
    newx = x*np.cos(rad) - y*np.sin(rad)
    newy = x*np.sin(rad) + y*np.cos(rad)
    new_poses = np.stack((newx,newy,z), axis=-1)
    return new_poses
@jit
def rotate_x(poses, rad):
    '''
    :poses:array,shape(...,3)
    :rad:float
    :ret:array,shape(...,3)
    '''
    x = poses[...,0]
    y = poses[...,1]
    z = poses[...,2]
    newy = y*np.cos(rad) - z*np.sin(rad)
    newz = y*np.sin(rad) + z*np.cos(rad)
    new_poses = np.stack((x,newy,newz), axis=-1)
    return new_poses

@jit
def align(head, neck, poses):
    '''
    :head:array,shape(3)
    :neck:array,shape(3)
    :poses:array,shape(n,3)
    :ret:array,shape(n,3)
    '''
    #translation
    neck = neck - head
    poses = poses - head
    head = head - head
    pos_shape = poses.shape##
    #rotate-z
    src_rad = np.arctan2(neck[1], neck[0])
    dst_rad = np.deg2rad(90)
    rot_rad = dst_rad - src_rad
    neck = rotate_z(neck, rot_rad)
    poses = rotate_z(poses, rot_rad)
    assert neck.shape == (3,)##
    assert poses.shape == pos_shape##
    #rotate-x
    src_rad = np.arctan2(neck[2], neck[1])
    dst_rad = np.deg2rad(-90)
    rot_rad = dst_rad - src_rad
    neck = rotate_x(neck, rot_rad)
    poses = rotate_x(poses, rot_rad)
    assert neck.shape == (3,)##
    assert poses.shape == pos_shape##
    return head, neck, poses

@jit
def dist_p2p(a, b, poses):
    '''
    :a,b:plane of ax+by=0
    :poses:array,(n,3)
    :ret:array,(n)
    '''
    x = poses[:,0]
    y = poses[:,1]
    dists = np.abs(a*x+b*y)/np.sqrt(a*a+b*b)
    assert dists.shape == (poses.shape[0],)
    return dists

@jit
def pos2p2d(a, b, poses):
    '''
    pos->proj_pos->2dpos

    :a,b:plane of ax+by=0
    :poses:array,(n,3)
    :ret:array,int,(n,2)
    '''
    #pos->proj:(b2x+aby)/(a2+b2),(a2y-abx)/(a2+b2)
    x = poses[:,0]
    y = poses[:,1]
    z = poses[:,2]
    proj_x = b*b*x - a*b*y
    proj_y = a*a*y - a*b*x
    proj_z = z
    #proj->2dpos:(sgn(x)*sqrt(x2+y2),z)
    x2d = np.copysign(np.sqrt(proj_x*proj_x+proj_y*proj_y), proj_x)
    y2d = proj_z
    p2ds = np.stack((x2d,y2d), axis=1)
    assert p2ds.shape == (poses.shape[0],2)
    return p2ds

@jit
def p2d2img(slice, xmax, xmin, ymax, ymin):
    '''
    2dpos->ix->img

    :slice:(n,2)
    :xmax,xmin,ymax,ymin:float
    :ret:(img_height,img_width)
    '''
    #2dpos->ix
    x = slice[:,0]
    y = slice[:,1]
    xpag = (x-xmin) / (xmax-xmin)
    #ixs = np.round(img_width*xpag).astype(np.int)
    ixs = (img_width*xpag).astype(np.int)
    ypag = (y-ymin) / (ymax-ymin)
    #iys = np.round(img_height*ypag).astype(np.int)
    iys = (img_height*ypag).astype(np.int)
    assert ixs.shape == (len(slice),)
    assert iys.shape == (len(slice),)
    #ix->img
    img = np.zeros((img_height,img_width))
    for ix,iy in zip(ixs,iys):
        img[iy,ix] += 1
    max_val = img.max()
    if max_val > 0:
        img = img / max_val * 255
    img = img.astype(np.uint8)
    img = np.flip(img, 0) #reverse
    assert img.shape == (img_height,img_width)
    return img

@jit
def make_slice(poses, rad):
    '''
    :poses:array,(n,3)
    :rad:float
    :ret:array,(m,2)
    '''
    #ax+by=0
    x0 = np.cos(rad)
    y0 = np.sin(rad)
    a = y0
    b = -x0

    dists = dist_p2p(a, b, poses)
    cond = dists < dist_threshold
    close_poses = np.compress(cond, poses, axis=0)

    assert close_poses.shape[1] == 3
    p2ds = pos2p2d(a, b, close_poses)
    assert p2ds.shape[1] == 2
    ##assert p2ds.shape[0] > 0
    return p2ds

#@jit
def calc_limit(slices):
    '''
    slices:list,[(n,2),(m,2)...]
    '''
    #get limit
    xmax = []
    xmin = []
    ymax = []
    ymin = []
    for slice in slices:
        if slice.shape[0] == 0:
            continue
        slicex = slice[:,0]
        slicey = slice[:,1]
        xmax.append(slicex.max())
        ymax.append(slicey.max())
        xmin.append(slicex.min())
        ymin.append(slicey.min())
    assert len(xmax) > 0
    xmax = max(xmax)
    ymax = max(ymax)
    xmin = min(xmin)
    ymin = min(ymin)
    if np.abs(np.array([xmax,xmin,ymax,ymin])).max() > 5:
        print("limit warning:",xmax,xmin,ymax,ymin, file=logfile, flush=True)
    #extend limit
    xlen = xmax - xmin
    dx = xlen * 0.025
    xmax += dx
    xmin -= dx
    ylen = ymax - ymin
    dy = ylen * 0.025
    ymax += dy
    ymin -= dy
    return xmax,xmin,ymax,ymin

def make_img(poses, sklt):
    '''
    :poses:array,(n,3)
    :sklt:tuple,((body,12),(body,1,1,1),(body,1,1,1))
    :ret:array,(img_height,img_width,channels)
    '''
    #check
    head_id = 3
    neck_id = 0
    head_name = sklt[1][head_id][0][0]
    neck_name = sklt[1][neck_id][0][0]
    head_status = sklt[2][head_id][0][0]
    neck_status = sklt[2][neck_id][0][0]
    assert head_name == 'Head'
    assert neck_name == 'HipCenter'
    if head_status != 'Tracked' or neck_status != 'Tracked':#error frame
        return
    #get head and neck
    head = sklt[0][head_id][4:7]
    neck = sklt[0][neck_id][4:7]
    #align
    head,neck,poses = align(head, neck, poses)
    #make slices
    slices = []
    for i in range(channels):
        deg = 180 / channels * i
        rad = np.deg2rad(deg)
        slices.append(make_slice(poses, rad))
    #calc max and min
    xmax,xmin,ymax,ymin = calc_limit(slices)
    #make imgs
    imgs = [p2d2img(slice,xmax,xmin,ymax,ymin) for slice in slices]
    img = np.stack(imgs, axis=-1)
    assert img.shape == (img_height,img_width,channels)
    return img

def switch_sklt(data2, sklt0):
    sklt = []
    for frame in sklt0:
        frame = data2[frame[0]]
        a0 = np.transpose(frame['skl_data'])
        a1 = np.transpose(frame['joint_name'])
        a2 = np.transpose(frame['joint_status'])
        a1 = [[[''.join(map(chr, data2[a[0]][:,0]))]] for a in a1]
        a2 = [[[''.join(map(chr, data2[a[0]][:,0]))]] for a in a2]
        sklt.append([a0,a1,a2])
    return sklt

def io_read(pc_path, sklt_path):
    try:
        data = sio.loadmat(pc_path)
        pc = data['pointcloudt'][0] #(frames)
        pc = pc / 1000 #mm->m
        data2 = sio.loadmat(sklt_path)
        sklt = data2['sklt'][0] #(frames)
    except NotImplementedError as e:
        print("warning:{} can't open by scipy.io.loadmat".format(pc_path), file=logfile, flush=True)
        data = h5py.File(pc_path)
        pc0 = data['pointcloudt']
        pc = np.array([np.transpose(data[frame[0]]) for frame in pc0])
        pc = pc / 1000 #mm->m
        data2 = h5py.File(sklt_path)
        sklt0 = data2['sklt']
        sklt = switch_sklt(data2, sklt0)

    return pc,sklt

def io_save(save_path, outputs):
    args = {'save_path':save_path,'outputs':outputs}
    while not saveThread.isOff():
        time.sleep(0.01)
    saveThread.setOn(args)

def work(filepath, output_path):
    '''
    :filepath:path of input
    :output_path:path of output
    '''
    #io-read
    pc_path = os.path.join(filepath, "pointcloud.mat")
    sklt_path = os.path.join(filepath, "skl.mat")
    pc,sklt = io_read(pc_path, sklt_path)
    num_frame = len(pc)
    print("input get {} frames...".format(num_frame), file=logfile, flush=True)
    #calc
    outputs = []
    for i in range(num_frame):
        if isinstance(sklt, list):
            output = make_img(pc[i], sklt[i]) #new version
        else:
            output = make_img(pc[i], np.asscalar(sklt[i][0][0])) #old version
        if output is not None:
            outputs.append(output)
    outputs = np.array(outputs)#(frames,img_height,img_width,channels)
    #io-write
    id_path = filepath.split('/')[-1]##
    output_dir = os.path.join(output_path, id_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outputfile_path = os.path.join(output_dir, 'slice{}.mat'.format(dist_threshold))
    io_save(outputfile_path, outputs)

    print("record {} get {} valid frames!".format(id_path,len(outputs)), file=logfile, flush=True)

def main(args):
    filelist = args.filelist
    output_path = args.output
    logpath = args.log

    global logfile
    logfile = open(os.path.join(logpath, 'log-'+filelist.split('/')[-1]), 'w')
    print("new run...", file=logfile, flush=True)

    with open(filelist) as f:
        filepaths = [line.strip() for line in f]
    for i,path in enumerate(filepaths):
        st_time = time.time()
        work(path, output_path)
        print("use time:{}".format(time.time()-st_time), file=logfile, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filelist', default='/home/jianheng/reid/preprocessing/filelist/filelist.txt', help='path of filelist')
    parser.add_argument('-o', '--output', default='/home/share/jianheng/reid/output/tmp', help='path of output')
    parser.add_argument('-l', '--log', default='/home/jianheng/reid/preprocessing/log', help='path of log')

    args = parser.parse_args()

    saveThread.start()

    try:
        main(args)
        #cProfile.run("main(args)", "result3.txt")
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("{}\n{}\n{}\n".format(exc_type,exc_value,traceback.format_exc()), file=logfile, flush=True)
    finally:
        time.sleep(5)
        saveThread.setExit()
        saveThread.join()
