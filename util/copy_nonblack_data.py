import shutil
import os
import glob
import cv2
import numpy as np

def get_nb_nonblack_pixels(img : np.array):
    img.reshape(-1)


def copy_nonblack_data(src_dir, dst_dir):
    """
    Copy all non-black images from src_dir to dst_dir.
    """
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for fname in glob.glob(os.path.join(src_dir, '*')):
        if os.path.isdir(fname):
            continue
        if os.path.getsize(fname) == 0:
            continue
        if os.path.basename(fname).startswith('.'):
            continue
        if os.path.splitext(fname)[1].lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        if os.path.getsize(fname) < 1000:
            continue
        try:
            with open(fname, 'rb') as f:
                data = f.read()
            if len(data) < 1000:
                continue
            with open(os.path.join(dst_dir, os.path.basename(fname)), 'wb') as f:
                f.write(data)
        except Exception as e:
            print(e)
            continue