'''
seeling:17.11.07

v-1.0
-auto add or remove files and dirs
'''

import os
import shutil
import argparse
import codecs

class Sync(object):
    def __init__(self, args):
        self.args = args
        self.add_files = []
        self.add_dirs = []
        self.del_files = []
        self.del_dirs = []

    def print_tmp(self):
        cnt = len(self.add_files)+len(self.add_dirs)+len(self.del_files)+len(self.del_dirs)
        print(cnt, end='\r')

    def add_dir(self, src, dst):
        os.mkdir(dst)
        self.add_dirs.append(src)
        self.print_tmp()

    def add_file(self, src, dst):
        shutil.copy(src, dst)
        self.add_files.append(src)
        self.print_tmp()

    def del_dir(self, src, dst):
        os.rmdir(dst)
        self.del_dirs.append(src)
        self.print_tmp()

    def del_file(self, src, dst):
        os.remove(dst)
        self.del_files.append(src)
        self.print_tmp()

    def dfs(self, path):
        src_names = set(os.listdir(os.path.join(self.args.src, path)))
        dst_names = set(os.listdir(os.path.join(self.args.dst, path)))

        add_names = src_names - dst_names
        del_names = dst_names - src_names

        for name in add_names:
            src_path = os.path.join(self.args.src, path, name)
            dst_path = os.path.join(self.args.dst, path, name)
            if os.path.isdir(src_path):
                self.add_dir(src_path, dst_path)
            else:
                self.add_file(src_path, dst_path)
        for name in del_names:
            src_path = os.path.join(self.args.src, path, name)
            dst_path = os.path.join(self.args.dst, path, name)
            if os.path.isdir(dst_path):
                self.dfs_del(os.path.join(path, name))
                self.del_dir(src_path, dst_path)
            else:
                self.del_file(src_path, dst_path)

        for name in os.listdir(os.path.join(self.args.src, path)):
            if os.path.isdir(os.path.join(self.args.src, path, name)):
                self.dfs(os.path.join(path, name))

    def dfs_del(self, path):
        for name in os.listdir(os.path.join(self.args.dst, path)):
            src_path = os.path.join(self.args.src, path, name)
            dst_path = os.path.join(self.args.dst, path, name)
            if os.path.isdir(dst_path):
                self.dfs_del(os.path.join(path, name))
                self.del_dir(src_path, dst_path)
            else:
                self.del_file(src_path, dst_path)

    def run(self):
        if not os.path.exists(self.args.dst):
            os.makedirs(self.args.dst)

        self.dfs('')

        print("total add {} files, delete {} files".format(len(self.add_files),len(self.del_files)))
        print("total add {} dirs, delete {} dirs".format(len(self.add_dirs),len(self.del_dirs)))
        with codecs.open('sync.log', 'w', 'utf-8') as f:
            for file in self.add_files:
                f.write("{}\n".format(file))
            f.write("\n")
            for file in self.del_files:
                f.write("{}\n".format(file))
            f.write("\n")
            for d in self.add_dirs:
                f.write("{}\n".format(d))
            f.write("\n")
            for d in self.del_dirs:
                f.write("{}\n".format(d))

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='src dir')
    parser.add_argument('--dst', help='dst dir')
    args = parser.parse_args()

    if args.src is None or args.dst is None:
        print("please key in 'python sync.py -h' for usage")
        exit(1)

    sync = Sync(args)
    sync.run()
