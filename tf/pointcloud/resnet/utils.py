import logging
import argparse

args = None

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir')
    parser.add_argument('--learn_rate', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--beg_epoch', default=0, type=int)
    parser.add_argument('--model_dir')
    parser.add_argument('--model_path')
    parser.add_argument('--reset_log', action='store_true')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--img_height', type=int)
    parser.add_argument('--img_width', type=int)
    parser.add_argument('--img_channels', type=int)
    parser.add_argument('--classes', default=333, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--gallery_dir')
    parser.add_argument('--probe_dir')
    parser.add_argument('--testid_path')
    parser.add_argument('--rad_sigma', default=0.531579584420809,type=float)

    global args
    args = parser.parse_args()

make_args()


log_filemode = 'w' if args.reset_log else 'a'

base_fh = logging.FileHandler('./log/all.log', mode=log_filemode)
base_fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s <line:%(lineno)d>:%(message)s", datefmt='%m-%d %H:%M:%S',))
base_fh.setLevel(logging.WARNING)

def make_logger(name, level=logging.INFO, is_stdout=True, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if is_stdout:
        fh = logging.StreamHandler()
    else:
        fh = logging.FileHandler(filename, mode=log_filemode)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s <line:%(lineno)d>:%(message)s", datefmt='%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(base_fh)
    return logger




