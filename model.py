# -*- coding: utf-8 -*-
'''
   model.py 文件
   利用slim模块的net进行对应的模型搭建
   包括定义loss，accaury以及优化算法的设计          
'''

import tensorflow as tf
from tensorflow.contrib.slim import nets
import preprocessing
slim = tf.contrib.slim
    
        
class Model(object):
    
    def __init__(self, is_training,
                 num_classes=2,
                 fixed_resize_side=256,
                 default_image_size=224):
        # model 模型初始化，传入的参数类型
        self._num_classes = num_classes
        self._is_training = is_training
        self._fixed_resize_side = fixed_resize_side
        self._default_image_size = default_image_size
        
    @property
    def num_classes(self):
        return self._num_classes
        
    def preprocess(self, inputs):
        """
        利用preprocessing.py中的数据处理函数对输入数据进行处理
        输入 [batch_size,height, width, num_channels]  a batch of images. 
        输出 tensors [batch_size,height, width, num_channels]
        """
        # 调用批处理函数进行处理
        preprocessed_inputs = preprocessing.preprocess_images(
            inputs, self._default_image_size, self._default_image_size, 
            resize_side_min=self._fixed_resize_side,
            is_training=self._is_training,
            border_expand=False, normalize=False,
            preserving_aspect_ratio_resize=False)
        # 转化数据类型为 float32
        preprocessed_inputs = tf.cast(preprocessed_inputs, tf.float32)
        return preprocessed_inputs
    
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        输入 [batch_size,height, width, num_channels]  a batch of images. 
        利用slim模块加载预训练模型，然后返回softmax之前的参数
        输出 预测值{'logits': logits}
        """

        # resnet_v1_50 函数返回的形状为 [None, 1, 1, num]，
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, endpoints = nets.resnet_v1.resnet_v1_50(
                preprocessed_inputs, num_classes=None,
                is_training=self._is_training)
        # 为了输入到全连接层，需要用函数 tf.squeeze 去掉形状为 1 的第 1，2 个索引维度。
        net = tf.squeeze(net, axis=[1, 2])
        # 将resnet的最后一层输出进行处理，变成二分类
        logits = slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, 
                                      scope='Predict/logits')
        return {'logits': logits}
    
    def postprocess(self, prediction_dict):
        # 返回结果dict
        postprocessed_dict = {}
        for logits_name, logits in prediction_dict.items():
            logits = tf.nn.softmax(logits)
            classes = tf.argmax(logits, axis=1)
            classes_name = logits_name.replace('logits', 'classes')
            postprocessed_dict[logits_name] = logits
            postprocessed_dict[classes_name] = classes
        return postprocessed_dict
    
    def loss(self, prediction_dict, groundtruth_lists):
        # logits，和 y 之间使用cross_entropy
        logits = prediction_dict.get('logits')
        slim.losses.sparse_softmax_cross_entropy(logits, groundtruth_lists)
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict
        
    def accuracy(self, postprocessed_dict, groundtruth_lists):
        # y_，和 y 之间计算accaury
        classes = postprocessed_dict['classes']
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(classes, groundtruth_lists), dtype=tf.float32))
        return accuracy

