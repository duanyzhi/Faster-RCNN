# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from lib.utils.cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN
    rpn_cls_score： [1, w, h, 9*2]  softmax之前的bg fg软值
    gt_boxes： 图片真实的box
    im_info：图片大小
    _feat_stride：16
    all_anchors： 输出的未经过处理的所有的原始anchor
    num_anchors： 每个特征点9个anchor
    输出: tx*  ty*  tw* th*
    """
    A = num_anchors    # 9
    total_anchors = all_anchors.shape[0]  # 所有anchor
    K = total_anchors / num_anchors
    im_info = im_info[0]  # 取一张图片大小

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    # only keep anchors inside the image   inds_inside: [ 1404  1413  1422 ..., 15669 15678 15687],假设长度len(inds_inside) = len_inds_inside
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]
    # keep only inside anchors  去除大于边界的anchor
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)  # 生成全是-1的列表

    # overlaps between the anchors and the gt boxes   overlaps：重叠，即计算的IOU的值
    # overlaps (ex, gt)  ex:shape: [len_inds_inside, 4]  gt:shape: [K, 5]  K表示一张图上有几个物体
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    # 输出：overlaps shape: [len_inds_inside, K] ,每一列表示len_inds_inside个anchor和目标Box的重叠度，K列为每一个目标都算一次
    argmax_overlaps = overlaps.argmax(axis=1)   # shape: len_inds_inside, axis=1按行选择每一行最大值的位置（K个值里最大的值位置）
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]   # 输出shape: len_inds_inside  把overlaps每行最大值的位置取出来
    gt_argmax_overlaps = overlaps.argmax(axis=0)    # shape: K  axis=0 按列取出每一列的最大值位置 一共K个值
    gt_max_overlaps = overlaps[gt_argmax_overlaps,   # shape: K 把每列最大值取出来
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]  # 返回最大值位置的位置的坐标，然后取[0]表示取哪一行  [1]表示那一列

    if not cfg.FLAGS.rpn_clobber_positives:   # 根据IOU设置背景区域
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        # IOU < 0.3    max_overlaps < cfg.FLAGS.rpn_negative_overlap是一个True or False的列表
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0   # 按行先把iou小的置零

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1    # 把K个列的最大值位置处变为1，这些box是可以的

    # fg label: above threshold IOU    IOU > 0.7
    labels[max_overlaps >= cfg.FLAGS.rpn_positive_overlap] = 1  # 在按照行把iou大的变为1

    if cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.FLAGS.rpn_fg_fraction * cfg.FLAGS.rpn_batchsize)  # 设置anchor最多个数
    fg_inds = np.where(labels == 1)[0]   # 取label为1的anchor
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.FLAGS.rpn_batchsize - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    # gt_boxes[argmax_overlaps, :]: 把gt_box从[K, 5]变为
    # argmax_overlaps: len_inds_inside的向量 每个元素在[0, K-1]之间，表示每一行的anchor哪个图像重合位置最大
    # gt_boxes: [K, 5]
    # gt_boxes[argmax_overlaps, :]：shape为 [len_inds_inside, 5]  按照argmax_overlaps来确定取出gt_boxes哪一行（即哪一个box）
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])   # t*

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.FLAGS.bbox_inside_weights)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.FLAGS.rpn_positive_weight < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.FLAGS.rpn_positive_weight > 0) &
                (cfg.FLAGS.rpn_positive_weight < 1))
        positive_weights = (cfg.FLAGS.rpn_positive_weight /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.FLAGS.rpn_positive_weight) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
