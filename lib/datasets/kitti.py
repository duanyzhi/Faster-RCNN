from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import random
import numpy as np
import tensorflow as tf
from lib.datasets.imdb import imdb
import lib.config.config as cfg
from DeepLearning.python import text_read
from DeepLearning.deep_learning import Batch_Normalization

class kitti(imdb):
    def __init__(self):
        imdb.__init__(self)
        self.__train_image_index = self._load_image_set_index(cfg.FLAGS.train_list)  # __两个下划线只能在kitti类下面调用
        self.__test_image_index = self._load_image_set_index(cfg.FLAGS.test_list)
        self.__val_image_index = self._load_image_set_index(cfg.FLAGS.val_list)
        self.count = {"train": 0, "test": 0, "val": 0}
        self.blobs = {}

    def __call__(self, pattern):
        if pattern == 'train':
            if self.count[pattern] + 1 > len(self.__train_image_index):
                self.count[pattern] = 0
                random.shuffle(self.__train_image_index)
            self.set_proposal_method(self.__train_image_index[self.count[pattern]*cfg.FLAGS.batch_size:
                                                                     (self.count[pattern] + 1)*cfg.FLAGS.batch_size])
            self.count[pattern] += 1

            im = self._image_roidb[0]['data']
            im = im[np.newaxis, ...]
            im = Batch_Normalization(im)
            im_info = np.zeros((cfg.FLAGS.batch_size, 3))
            im_info[0, 0] = im.shape[1]
            im_info[0, 1] = im.shape[2]
            im_info[0, 2] = im.shape[3]
            return {'data': im,                               # [None, ?, ?, 3]  这里None是batch size
                    'boxes': self._image_roidb[0]['boxes'],   # [None, 5]   这里None是一张图有几个box
                    'img_name': self._image_roidb[0]['img_name'],      # [None]   none表示几个物体
                    'gt_classes': self._image_roidb[0]['gt_classes'],    # [None]  None表示几个物体的类别，是一个数0...8
                     'gt_overlaps': self._image_roidb[0]['gt_overlaps'],   # [None, 9]  one hot的标签
                     '_im_info': im_info,                            # [None, 3]  图片大小
                    'im_name':  self._image_roidb[0]['im_name']}

    @staticmethod
    def _load_image_set_index(file_name):
        return text_read(file_name)
