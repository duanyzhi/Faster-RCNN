import tensorflow as tf
import lib.config.config as cfg
from DeepLearning.deep_tensorflow import *
from lib.utils.resnet_utils import create_variables, batch_norm
import tensorflow.contrib.slim as slim



def Residual_block(in_img, filters, projection=False):
    input_depth = int(in_img.get_shape()[3])

    with tf.variable_scope('conv1'):  # 1*1
        weight1 = create_variables(name='weight_1', shape=[1, 1, input_depth, filters[0]])
        biases1 = create_variables(name='biases_1', shape=[filters[0]], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(in_img, weight1, strides=[1, 1, 1, 1], padding='SAME') + biases1
        bn = batch_norm('bn', conv)
        relu = tf.nn.relu(bn)

    with tf.variable_scope('conv2'):  # 3*3
        weight2 = create_variables(name='weight_2', shape=[3, 3, filters[0], filters[1]])
        biases2 = create_variables(name='biases_2', shape=[filters[1]], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(relu, weight2, strides=[1, 1, 1, 1], padding='SAME') + biases2
        bn = batch_norm('bn', conv)
        relu = tf.nn.relu(bn)
        # conv2_ReLu = batch_norm('bn', conv2_ReLu)
        # conv2_ReLu = Conv_BN_Relu(conv1_ReLu, weight2, biases2, filters[1], strides=1)

    with tf.variable_scope('conv3'):  # 1*1
        weight3 = create_variables(name='weight_3', shape=[1, 1, filters[1], filters[2]])
        biases3 = create_variables(name='biases_3', shape=[filters[2]], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(relu, weight3, strides=[1, 1, 1, 1], padding='SAME') + biases3
        out = batch_norm('bn', conv)
        # conv3_ReLu = Conv_BN_Relu(conv2_ReLu, weight3, biases3, filters[2], strides=1)

    if input_depth != filters[2]:
        if projection:
            # Option B: Projection shortcut
            weight_4 = create_variables(name='weight_4', shape=[1, 1, input_depth, filters[2]])
            biases_4 = create_variables(name='biases_4', shape=[filters[2]], initializer=tf.zeros_initializer())
            input_layer = Conv_BN_Relu(in_img, weight_4, biases=biases_4,  dimension=filters[2], strides=1)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(in_img, [[0, 0], [0, 0], [0, 0], [int((filters[2] - input_depth)/2), filters[2] - input_depth - int((filters[2] - input_depth)/2)]])  # 维度是4维[batch_size, :, :, dim] 我么要pad dim的维度
    else:
        input_layer = in_img

    output = out + input_layer
    output = tf.nn.relu(output)
    return output


class vgg:
    def __init__(self):
        self.img = None
        self.reuse = False
        self.learning_rate = cfg.FLAGS.learning_rate

    def __call__(self, img, scope):
        self.img = img         # [224, 224, 3]
        # Main network
        # Layer  1
        net = slim.repeat(self.img, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=cfg.FLAGS.is_training, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=cfg.FLAGS.is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=cfg.FLAGS.is_training, scope='conv5')
        # with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
        #     if self.reuse:
        #         scope_name.reuse_variables()
        #     # conv1
        #     self.img = tf.pad(self.img, [[0, 0], [1, 1], [1, 1], [0, 0]])
        #     with tf.variable_scope('conv1'):
        #         weight_1 = create_variables(name='weight', shape=[7, 7, 3, 64])
        #         biases_1 = create_variables(name='biases', shape=[64], initializer=tf.zeros_initializer())
        #         conv1_ReLu = tf.nn.conv2d(self.img, weight_1, strides=[1, 2, 2, 1], padding='SAME') + biases_1
        #         conv1_BN = batch_norm('bn', conv1_ReLu)
        #         conv1 = tf.nn.relu(conv1_BN)
        #         conv1 = max_pool(conv1, k_size=(2, 2), stride=(2, 2), pad='SAME')   # out [56, 56, 64]
        #     in_img = conv1
        #     # conv2
        #     with tf.variable_scope('conv2'):
        #         for kk in range(3):
        #             with tf.variable_scope('Residual_' + str(kk)):
        #                 in_img = Residual_block(in_img, [64, 64, 256])    # [56, 56, 256]
        #     conv2 = max_pool(in_img, k_size=(2, 2), stride=(2, 2), pad='SAME')    # [28, 28, 256]
        #     in_img = conv2
        #     with tf.variable_scope('conv3'):
        #         for kk in range(4):
        #             with tf.variable_scope('Residual_' + str(kk)):
        #                 in_img = Residual_block(in_img, [128, 128, 512])  # [28, 28, 512]
        #     conv3 = max_pool(in_img, k_size=(2, 2), stride=(2, 2), pad='SAME')    # [14, 14, 512]
        #     in_img = conv3
        #     with tf.variable_scope('conv4'):
        #         for kk in range(23):
        #             with tf.variable_scope('Residual_' + str(kk)):
        #                 in_img = Residual_block(in_img, [256, 256, 512])  # [14, 14, 1024]
        #     conv4 = max_pool(in_img, k_size=(2, 2), stride=(2, 2), pad='SAME')    # [7, 7, 1024]
        #     in_img = conv4
        #     with tf.variable_scope('conv5'):
        #         for kk in range(3):
        #             with tf.variable_scope('Residual_' + str(kk)):
        #                 in_img = Residual_block(in_img, [512, 512, 512])  # [7, 7, 2048]
        #     print(in_img)
        #     out_conv = in_img
        self.reuse = True
        return net


class rpn(object):
    def __init__(self):
        self.reuse = False
        # select initializer
        if cfg.FLAGS.initializer == "truncated":
            self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            self.initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            self.initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        self._num_anchors = 9   # 每个点9个anchor

    def rpn_cnn(self, feature_map, scope):
        '''
        feature_map:我们的特征图，假设宽w, 高h,则其大小[1, w, h, 512]
        '''
        with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
            if self.reuse:
                scope_name.reuse_variables()
            # RPN
            # 先对每个点做一个3*3卷积（即用3*3的窗划过所有特征点） rpn: [1, ?, ？， 512]
            rpn = slim.conv2d(feature_map, 512, [3, 3], trainable=cfg.FLAGS.is_training, weights_initializer=self.initializer,
                              scope="rpn_conv/3x3")

            # -----------------------------------------------------------------------------------------------
            # cls分类
            # 每个点经过3*3卷积核之后在通过1*1卷积核变为一个9*2的特征向量，用于表示这个anchor是目标或者背景
            # 输出rpn_cls_score大小： (1, ?, ?, 18)
            rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=cfg.FLAGS.is_training,
                                        weights_initializer=self.initializer, padding='VALID', activation_fn=None,
                                        scope='rpn_cls_score')
            # Change it so that the score has 2 as its channel size
            # reshape变为 [1, ?, ?, 2]
            # 过程： (1, w, h, 18) --> (1, 18, w, h) --> (1, 2, 9*w, h) -- > (1, 9*w, h, 2)
            rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
            # softmax输出分类    rpn_cls_prob_reshape输出shape (1, 9*w, h, 2)
            rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
            # reshape回去，输出[1, 9*2, w, h]
            # 过程： （1， 9*w, h, 2） --> （1， 2, 9*w, h） --> （1， 18, w, h）--> （1, w, h, 18）
            rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")

            # -----------------------------------------------------------------------------------------------
            # bbox
            # 卷积输出[1, w, h, 4*9]用于输出box的四个偏值 ，进行第一次微调
            rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=cfg.FLAGS.is_training,
                                        weights_initializer=self.initializer, padding='SAME', activation_fn=None,
                                        scope='rpn_bbox_pred')
            # 输出：
            '''
            rpn_cls_prob: [1, w, h, 9*2]  bg或者fg的概率
            rpn_bbox_pred： [1, w, h, 4*9]   4分别表示[tx, ty, tw, th]
            rpn_cls_score： [1, w, h, 9*2]  softmax之前的bg fg软值
            rpn_cls_score_reshape： [1, 9*w, h, 2]   softmax  之前reshape之后的软值
            '''
            return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    @staticmethod
    def _reshape_layer(bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name):
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[cfg.FLAGS.batch_size], [num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    @staticmethod
    def _softmax_layer(bottom, name):
        if name == 'rpn_cls_prob_reshape':
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

