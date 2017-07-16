import numpy as np
import tensorflow as tf
from .utils import get_shape, inst_norm, lkrelu

# 9-ResBlock Generator
class Generator(object):
    def __init__(self, name, inputs, ochan, stddev=0.02, center=True, scale=True, reuse=None):
        self._stddev = stddev
        self._ochan = ochan
        with tf.variable_scope(name, initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=reuse):
            self._inputs = inputs
            self._resnet = self._build_resnet(self._inputs)

    def __getitem__(self, key):
        return self._resnet[key]

    def _build_conv_layer(self, name, inputs, k, rfsize, stride, use_in=True, f=tf.nn.relu, reflect=False):
        layer = dict()
        with tf.variable_scope(name):
            layer['filters'] = tf.get_variable('filters', [rfsize, rfsize, get_shape(inputs)[-1], k])

            if reflect:
                # pad with 3, not indicated by the paper but torch CycleGAN does it this way
                layer['conv'] = tf.nn.conv2d(tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'), layer['filters'], strides=[1, stride, stride, 1], padding='VALID')
            else:
                layer['conv'] = tf.nn.conv2d(inputs, layer['filters'], strides=[1, stride, stride, 1], padding='SAME')
            layer['bn'] = inst_norm(layer['conv']) if use_in else layer['conv']
            layer['fmap'] = f(layer['bn'])
        return layer

    def _build_residual_layer(self, name, inputs, k, rfsize, blocksize=2, stride=1): # rfsize: receptive field size
        layer = dict()
        with tf.variable_scope(name):
            with tf.variable_scope('layer1'):
                layer['filters1'] = tf.get_variable('filters1', [rfsize, rfsize, get_shape(inputs)[-1], k])
                layer['conv1'] = tf.nn.conv2d(tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'), layer['filters1'], strides=[1, stride, stride, 1], padding='VALID')
                layer['bn1'] = inst_norm(layer['conv1'])
                layer['fmap1'] = tf.nn.relu(layer['bn1'])

            with tf.variable_scope('layer2'):
                layer['filters2'] = tf.get_variable('filters2', [rfsize, rfsize, get_shape(inputs)[-1], k])
                layer['conv2'] = tf.nn.conv2d(tf.pad(layer['fmap1'], [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'), layer['filters2'], strides=[1, stride, stride, 1], padding='VALID')
                layer['bn2'] = inst_norm(layer['conv2'])

            # No ReLu here (following http://torch.ch/blog/2016/02/04/resnets.html, as indicated by the authors)
            layer['fmap2'] = layer['bn2'] + inputs
        return layer

    def _build_deconv_layer(self, name, inputs, k, output_shape, rfsize): # fractional-strided conv layer
        layer = dict()

        with tf.variable_scope(name):
            output_shape = [tf.shape(inputs)[0]] + output_shape
            layer['filters'] = tf.get_variable('filters', [rfsize, rfsize, output_shape[-1], get_shape(inputs)[-1]])
            layer['conv'] = tf.nn.conv2d_transpose(inputs, layer['filters'], output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')
            layer['bn'] = inst_norm(tf.reshape(layer['conv'], output_shape))
            layer['fmap'] = tf.nn.relu(layer['bn'])
        return layer

    def _build_resnet(self, inputs):
        resnet = dict()

        inputs_shape = get_shape(inputs)
        width = inputs_shape[1]
        height = inputs_shape[2]

        # c7s1-32,d64,d128,R128,R128,R128,R128,R128,R128,R128,R128,R128,u64,u32,c7s1-3 See paper §7.5
        with tf.variable_scope('resnet'):
            resnet['l1'] = self._build_conv_layer('c7s1-32_1', inputs, k=32, rfsize=7, stride=1, reflect=True)
            resnet['l2'] = self._build_conv_layer('d64_1', resnet['l1']['fmap'], k=64, rfsize=3, stride=2)
            resnet['l3'] = self._build_conv_layer('d128_1', resnet['l2']['fmap'], k=128, rfsize=3, stride=2)
            resnet['l4'] = self._build_residual_layer('r128_1', resnet['l3']['fmap'], k=128, rfsize=3, stride=1)
            resnet['l5'] = self._build_residual_layer('r128_2', resnet['l4']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l6'] = self._build_residual_layer('r128_3', resnet['l5']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l7'] = self._build_residual_layer('r128_4', resnet['l6']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l8'] = self._build_residual_layer('r128_5', resnet['l7']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l9'] = self._build_residual_layer('r128_6', resnet['l8']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l10'] = self._build_residual_layer('r128_7', resnet['l9']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l11'] = self._build_residual_layer('r128_8', resnet['l10']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l12'] = self._build_residual_layer('r128_9', resnet['l11']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l13'] = self._build_deconv_layer('u64_1', resnet['l12']['fmap2'], k=64, output_shape=[width//2, height//2, 64], rfsize=3)
            resnet['l14'] = self._build_deconv_layer('u32_1', resnet['l13']['fmap'], k=32, output_shape=[width, height, 32], rfsize=3)
            resnet['l15'] = self._build_conv_layer('c7s1-3_1', resnet['l14']['fmap'], f=tf.nn.tanh, k=get_shape(inputs)[-1], rfsize=7, stride=1, use_in=False, reflect=True)
        return resnet
