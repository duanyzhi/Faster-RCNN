import os
import cv2
import numpy as np
import lib.config.config as cfg
import xml.etree.ElementTree as ET

class imdb(object):
    def __init__(self):
        self._image_roidb = []
        pass

    @property   # 定义静态方法（即方法里没用到类里其他方法或者self定义），将方法变为类。
    def append_flipped_images(self):
        return self._image_roidb

    '''
    self._image_roidb用来存储一个batch size的所有信息，包括img读的矩阵，bbox，cls
    '''
    @append_flipped_images.setter
    def append_flipped_images(self, img_name_list):
        for img_name in img_name_list:   # 对一个batch 分别处理
            im = cv2.imread(os.path.join(cfg.FLAGS.train_img, img_name + '.png'))  # 读图片
            boxes, name, gt_classes, overlaps, im_name = self._load_pascal_annotation(img_name)  # 加载标签
            self._image_roidb.append({'data': im,
                                 'boxes': boxes,
                                 'img_name': name,
                                 'gt_classes': gt_classes,
                                 'gt_overlaps': overlaps,
                                  'im_name': im_name})


    def set_proposal_method(self, method):   # method传入一个batch size的图像名称序列
        self._image_roidb = []
        self.append_flipped_images = method   # 调用append_flipped_images

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(cfg.FLAGS.train_label, index + '.xml')
        tree = ET.parse(filename)

        im_name = tree.findall('filename')[0].text

        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 5), dtype=np.uint16)   # 物体边界加类别
        gt_classes = np.zeros((num_objs), dtype=np.int32)  # 类别
        overlaps = np.zeros((num_objs, cfg.FLAGS.classes_numbers), dtype=np.float32)

        name = []
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):   #  对一张图上多个类别分别处理
            name.append(obj.find('name').text.strip())
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1   # 更改的地方  这里不能-1 因为kitti里最小就是0了 -1就不对了
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = cfg.FLAGS.KITTI_classes.index(obj.find('name').text.strip())
            boxes[ix, :] = [x1, y1, x2, y2, cls]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        return boxes, name, gt_classes, overlaps, im_name

