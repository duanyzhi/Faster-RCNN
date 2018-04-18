import tensorflow as tf
import lib.config.config as cfg


class Loss(object):
    def __init__(self, _predictions, _anchor_targets, _proposal_targets):
        self._tag = None
        self._predictions = _predictions
        self._anchor_targets = _anchor_targets
        self._proposal_targets = _proposal_targets
        self._num_classes = cfg.FLAGS.classes_numbers
        self._losses = {}
        pass

    @staticmethod
    def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss'):
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            self._losses['total_loss'] = loss
        return loss

    @staticmethod
    def optimizer(loss):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(loss, trainable_variables)

        if cfg.FLAGS.optimizer_name == "Mom":  # Momentum
            optimizer = tf.train.MomentumOptimizer(cfg.FLAGS.learning_rate, 0.9)
        elif cfg.FLAGS.optimizer_name == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
        elif cfg.FLAGS.optimizer_name == 'ADAM':
            optimizer = tf.train.AdamOptimizer(1e-4)
        else:  # SGD
            optimizer = tf.train.GradientDescentOptimizer(cfg.FLAGS.learning_rate)

        # 生成一个变量用于保存全局训练步骤( global training step )的数值,并使用 minimize() 函数更新系统中的三角权重( triangle weights )、增加全局步骤的操作
        # global_step定义存储训练次数，从1开始自己随着训练次数增加而增加。因此这个变量不需要训练的
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # train_step = optimizer.minimize(loss, global_step=global_step)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=global_step, name='train_step')

        train_ops = [apply_op]
        train_step = tf.group(*train_ops)
        return train_step
