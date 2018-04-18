import tensorflow as tf
import platform

FLAGS = tf.app.flags.FLAGS

# classes and color
tf.app.flags.DEFINE_string('KITTI_classes', ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'], 'kitti classes')
# color_list = ['red', 'blue', 'pink', 'yellow', 'green', 'gold', 'pink', 'gray', 'purple', 'yellow]
RGB_list = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 255, 0), (244, 164, 96), (139, 101, 118),
            (207, 207, 207), (0, 255, 255)]

# Kitti2012 data folder
if platform.system() == 'Windows':  # windows:
    tf.app.flags.DEFINE_string('train_img', 'E:\\KITTI2012\\PNGImages', 'KITTI2012 Image')
    tf.app.flags.DEFINE_string('train_label', 'E:\\KITTI2012\\Annotations', "KITTI2012 Label")
    tf.app.flags.DEFINE_string('train_list', 'E:\\KITTI2012\\ImageSets\\Main\\train.txt', 'the list of image name for train')
    tf.app.flags.DEFINE_string('test_list', 'E:\\KITTI2012\\ImageSets\\Main\\test.txt', 'the list of image name for test')
    tf.app.flags.DEFINE_string('val_list', 'E:\\KITTI2012\\ImageSets\\Main\\val.txt', 'the list of image name for val')
else:  # ubuntu
    tf.app.flags.DEFINE_string('train_img', '/media/dyz/Data/KITTI2012/PNGImages', 'KITTI2012 Image')
    tf.app.flags.DEFINE_string('train_label', '/media/dyz/Data/KITTI2012/Annotations', "KITTI2012 Label")
    tf.app.flags.DEFINE_string('train_list', '/media/dyz/Data/KITTI2012/ImageSets/Main/train.txt',
                               'the list of image name for train')
    tf.app.flags.DEFINE_string('test_list', '/media/dyz/Data/KITTI2012/ImageSets/Main/test.txt',
                               'the list of image name for test')
    tf.app.flags.DEFINE_string('val_list', '/media/dyz/Data/KITTI2012/ImageSets/Main/val.txt', 'the list of image name for val')

# model parameters
tf.app.flags.DEFINE_integer('batch_size', 1, "Network batch size during training")
tf.app.flags.DEFINE_integer('max_iter', 10000000, "Max iteration")
tf.app.flags.DEFINE_integer('step_size', 10000, "Step size for reducing the learning rate, currently only support one step")
tf.app.flags.DEFINE_integer('display', 10, "Iteration intervals for showing the loss during training, on command line interface")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")
tf.app.flags.DEFINE_float('classes_numbers', 9, "len of classes in KITTI2012")
tf.app.flags.DEFINE_string('is_training', True, 'default is train, check if train or test ')
tf.app.flags.DEFINE_integer('extra_train_ops',  [], 'for bn')
tf.app.flags.DEFINE_string('initializer', 'truncated', 'weight init model')
tf.app.flags.DEFINE_string('optimizer_name', 'Mom', 'train model')


# pre model
tf.app.flags.DEFINE_string('resnet_ckpt', 'data/pre_ckpt/resnet_v1_101.ckpt', 'pre training model on resnet 101')
tf.app.flags.DEFINE_string('bbox_inside_weights', (1.0, 1.0, 1.0, 1.0), 'bbox_inside_weights')
tf.app.flags.DEFINE_string('bbox_normalize_means', (0.0, 0.0, 0.0, 0.0), 'bbox_inside_weights')
tf.app.flags.DEFINE_string('bbox_normalize_stds', (0.1, 0.1, 0.1, 0.1), 'bbox_inside_weights')

# Testing Parameters #
######################
tf.app.flags.DEFINE_string('test_mode', "top", "Test mode for bbox proposal")  # nms, top


# RPN
tf.app.flags.DEFINE_integer('rpn_train_pre_nms_top_n', 12000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_train_post_nms_top_n', 2000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_pre_nms_top_n', 6000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_post_nms_top_n', 300, "Number of top scoring boxes to keep before apply NMS to RPN proposals")

tf.app.flags.DEFINE_float('rpn_negative_overlap', 0.3, "IOU < thresh: negative example")
tf.app.flags.DEFINE_float('rpn_positive_overlap', 0.7, "IOU >= thresh: positive example")
tf.app.flags.DEFINE_float('rpn_fg_fraction', 0.5, "Max number of foreground examples")
tf.app.flags.DEFINE_float('rpn_train_nms_thresh', 0.7, "NMS threshold used on RPN proposals")
tf.app.flags.DEFINE_float('rpn_test_nms_thresh', 0.7, "NMS threshold used on RPN proposals")
tf.app.flags.DEFINE_integer('roi_pooling_size', 7, "Size of the pooled region after RoI pooling")
tf.app.flags.DEFINE_boolean('rpn_clobber_positives', False, "If an anchor satisfied by positive and negative conditions set to negative")
tf.app.flags.DEFINE_integer('rpn_batchsize', 256, "Total number of examples")
tf.app.flags.DEFINE_integer('rpn_positive_weight', -1,
                            'Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p).'
                            'Set to -1.0 to use uniform example weighting')

tf.app.flags.DEFINE_float('proposal_fg_fraction', 0.25, "Fraction of minibatch that is labeled foreground (i.e. class > 0)")
tf.app.flags.DEFINE_boolean('proposal_use_gt', False, "Whether to add ground truth boxes to the pool when sampling regions")
tf.app.flags.DEFINE_float('roi_fg_threshold', 0.5, "Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)")
tf.app.flags.DEFINE_float('roi_bg_threshold_high', 0.5, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")
tf.app.flags.DEFINE_float('roi_bg_threshold_low', 0.01, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")

tf.app.flags.DEFINE_boolean('bbox_normalize_targets_precomputed', True, "# Normalize the targets using 'precomputed' (or made up) means and stdevs (BBOX_NORMALIZE_TARGETS must also be True)")
tf.app.flags.DEFINE_boolean('test_bbox_reg', True, "Test using bounding-box regressors")



