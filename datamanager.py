# coding:utf-8

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import scipy.io as sio


class Dataset(object):

    def __init__(self, normalize=True, img=None):

        print("\nInitializing Dataset...")
        self.normalize = normalize
        self.img = img
        row, col, channel = self.img.shape
        img = np.reshape(img, (row * col, channel))

        print('data shape: ', img.shape)
        self.x_tr, self.y_tr = img, img,
        self.x_te, self.y_te = img, img,

        self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
        self.x_te = np.ndarray.astype(self.x_te, np.float32)
        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        print("Number of data\nTraining: %d, Test: %d\n" % (self.num_tr, self.num_te))
        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.min_val, self.max_val = x_sample.min(), x_sample.max()

    def reset_idx(self):
        self.idx_tr, self.idx_te = 0, 0

    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr + batch_size
        x_tr, y_tr = self.x_tr[start:end, :], self.y_tr[start:end, :]
        x_tr = np.array(x_tr)
        terminator = False

        if end >= self.num_tr:
            terminator = True
            self.idx_tr = 0
            self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else:
            self.idx_tr = end

        if fix:
            self.idx_tr = start

        if self.normalize:
            min_x, max_x = np.min(x_tr), np.max(x_tr)
            x_tr = (x_tr - min_x) / (max_x - min_x)

        return x_tr, y_tr, terminator

    def next_test(self, batch_size=32):

        start, end = self.idx_te, self.idx_te + batch_size
        x_te, y_te = self.x_te[start:end, :], self.y_te[start:end, :]
        x_te = np.array(x_te)
        terminator = False
        if end >= self.num_te:
            terminator = True
            self.idx_te = 0
        else:
            self.idx_te = end

        if x_te.shape[0] != batch_size:
            x_te = self.x_te[-1 - batch_size:-1, :]
            x_te = np.array(x_te)

        if self.normalize:
            min_x, max_x = np.min(x_te), np.max(x_te)
            x_te = (x_te - min_x) / (max_x - min_x)

        return x_te, y_te, terminator









