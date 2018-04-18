# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
        :param anchors:  所有anchor  [?, 4]
        :param _gt_boxes:  ground true box   [？， 5]
        :param rpn_bbox_pred:  rpn输出的 [1, w, h, 4*9]   4分别表示[tx, ty, tw, th]
        :param rpn_cls_prob:  fg bg 概率  [1, w, h, 2*9]
        :param im_info : [None, 3]图片大小
        :param cfg_key tran or test model
        :param num_anchors: 9  一个点的anchor个数
        :return:
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == "train":
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh

    im_info = im_info[0]
    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]      # 令前九个和后九个组成九个概率分布，我们取后九个表示fg的概率，值越大表示是物体概率越大。一共w*h*9个。和anchors个数一样
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))    # 取出dx dy dw dh， 一共w*h*9个1*4的向量。和anchors一样数量
    scores = scores.reshape((-1, 1))
    # print(anchors, "\r\n", "rpn", rpn_bbox_pred)
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)   # proposals表示输出修正后的bbox
    proposals = clip_boxes(proposals, im_info[:2])           # 剔除大于边界的情况，将大于边界的box换为边界值

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]  # ravel():行序列优先将矩阵展开为向量  .argsort()：将向量元素从小到大排序，提取对应index返回，返回的一个数是最小值的位置索引  [::-1]取反
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]   # 取前topN个
    proposals = proposals[order, :]   # 取出对应的box
    scores = scores[order]           # 取出对应的sacore

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]   # 取前2000个
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob, scores   # blob： [2000, 5]
