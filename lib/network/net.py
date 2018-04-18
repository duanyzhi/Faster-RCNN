from lib.datasets.kitti import kitti
import numpy as np
data = kitti()
import tensorflow as tf

from lib.network.cnn import *
import lib.config.config as cfg
from lib.utils.generate_anchors_pre import generate_anchors_pre
from lib.utils.smooth_L1_loss import smooth
from lib.utils.nms import nms
from .proposals import proposal
from .loss import Loss
from .draw import draw_demo

cnn = vgg()
rpn = rpn()
loss_function = smooth()
draw = draw_demo()


class net(proposal):
    def __init__(self, pattern):
        self.pattern = pattern
        if self.pattern == 'train':
            cfg.FLAGS.is_training = True
        else:
            cfg.FLAGS.is_training = False
        self._feat_stride = [16.0, 16.0]  # 原始图片是卷积后特征图片（RPN之前）大小的31倍  [宽的倍数， 高的倍数]
        self._anchor_scales = (8, 16, 32)   # 默认的anchor生成比例
        self._anchor_ratios = (0.5, 1, 2)
        self._num_anchors = 9  # 每个点9个anchor
        self._anchor_targets = {}
        self._proposal_targets = {}
        self.placeholder()
        self._predictions = {}
        proposal.__init__(self, self._im_info, self.pattern, self._feat_stride, self._num_anchors, self._gt_boxes)

    def placeholder(self):
        self.img = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[cfg.FLAGS.batch_size, 3])  # 表示图片的三维尺寸
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])   # 用于表示[xmin, xmax, ymin, ymax, cls] ground true

    def build_faster_rcnn(self):
        # 第一步图片经过cnn提取特征feature map
        cnn_conv = cnn(self.img, 'resnet_rpn')

        # 对残生的特征图每一个点9个anchor生成所有的anchor
        self._get_anchor()

        # 对特征图进行并行的两路卷积，分别生成rpn的cls和box的修正值[dx, dy, dw, dh]
        rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = rpn.rpn_cnn(cnn_conv, 'RPN')

        # Build proposal输出
        rois = self.build_proposals(cfg.FLAGS.is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, self._anchors)

        # Build predictions
        cls_score, cls_prob, bbox_pred = self.build_predictions(cnn_conv, rois, cfg.FLAGS.is_training, self.initializer, self.initializer_bbox)

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions["rois"] = rois

        _loss = Loss(self._predictions, self._anchor_targets, self._proposal_targets)
        loss = _loss.add_losses()

        train_step = _loss.optimizer(loss)

        return train_step, loss

    def gpu_config(self):
        self.saver = tf.train.Saver()
        # 下面设置GPU分配方式可以有效避免出现Resource exhausted: OOM when allocating tensor with...等内存不足的情况
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)  # 最多占用GPU 70%资源
        #  开始不会给tensorflow全部gpu资源 而是按需增加
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        if self.pattern == 'train':    # 只有训练模式才对变量初始化
            sess.run(tf.global_variables_initializer())
        return sess

    def _get_anchor(self):   # 根据特征图生成对应的anchor
        # 这里要先计算特征图片大小，因为tensorflow里面宽 长输入时一直变的 但是呢 卷积后大小是原始图片1/31。所以利用原始
        # 图片算出特征图长宽
        height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
        width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))
        # anchor是所有的anchors， 一个w*h*9个anchor
        anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                            [height, width,
                                             self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                            [tf.float32, tf.int32], name="generate_anchors")
        anchors.set_shape([None, 4])
        anchor_length.set_shape([])
        self._anchors = anchors   # 所有生成的anchor
        self._anchor_length = anchor_length

    def train(self, sess, train_step, loss):
        for kk in range(cfg.FLAGS.max_iter):
            roidb = data("train")
            _train_step, _loss, pre = sess.run([train_step, loss, self._predictions], feed_dict={self.img: roidb['data'],
                                                     self._im_info: roidb['_im_info'],
                                                     self._gt_boxes: roidb['boxes']})
            print(kk, "loss", _loss)

            if kk % 1000 == 0:
                box = pre["bbox_pred"]
                scores = pre["cls_prob"]
                im = roidb['data'][0, ...]
                img_name = roidb['im_name']
                print(img_name)
                draw.draw_box(im, box, scores, img_name)
            if kk % 1000 == 0:
                self.saver.save(sess, "data/ckpt/" + str(kk) + "_model.ckpt")




