import matplotlib.pyplot as plt
import lib.config.config as cfg
import numpy as np
from lib.utils.nms import nms
import cv2
from DeepLearning.Image import draw_box

class draw_demo():
    def draw_box(self, im, boxes, scores, img_name):
        print(boxes.shape, scores.shape)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for cls_ind, cls in enumerate(cfg.FLAGS.KITTI_classes):
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, 0.01)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= 0.5)[0]
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                if score > 0.2:
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]),
                                      bbox[2] - bbox[0],
                                      bbox[3] - bbox[1], fill=False,
                                      edgecolor='red', linewidth=3.5)
                    )
                    ax.text(bbox[0], bbox[1] - 2,
                            '{:s} {:.3f}'.format(cfg.FLAGS.KITTI_classes[cls_ind], score),
                            bbox=dict(facecolor='blue', alpha=0.5),
                            fontsize=14, color='white')

                else:
                    pass
            cls_ind += 1  # because we skipped background
        # plt.show()
        plt.savefig('data/PNG_out/' + img_name)

