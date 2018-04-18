from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.generate_anchors import generate_anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
      height:RPN特征图片大小  12
      width:RPN特征图片宽   39
      feat_stride:原始图片是特征图片的feat_stride倍
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride[0]
    shift_y = np.arange(0, height) * feat_stride[1]
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])
    # length表示一共多少个anchors， length = 卷积后的特征图片宽*卷积后的特征图片长*9
    return anchors, length
