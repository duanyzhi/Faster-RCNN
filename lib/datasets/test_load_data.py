from lib.datasets.kitti import kitti
import cv2
import numpy as np
from DeepLearning.Image import draw_box
data = kitti()

while True:
    roidb = data("train")

    for kk in range(len(roidb)):
        im = roidb[kk]['data']
        len_box = roidb[kk]['boxes'].shape[0]
        for jj in range(len_box):
            im = draw_box(im, roidb[kk]['boxes'][jj, ...], color_box=(0, 0, 0), thick_bbox=2, thick_circle=8)
            print(roidb[kk]['gt_overlaps'][jj, ...])
        cv2.imshow('t', im)
        cv2.waitKey()