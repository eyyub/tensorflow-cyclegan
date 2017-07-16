import numpy as np
import tensorflow as tf
from .utils import get_shape, inst_norm, lkrelu

# PatchGAN
class Discriminator(object):
    def __init__(self, name, inputs, stddev=0.02, reuse=None):
        self._stddev = stddev

        with tf.variable_scope(name, initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=reuse):
            self._inputs = inputs
            self._discriminator = self._build_discriminator(inputs)

    def __getitem__(self, key):
        return self._discriminator[key]

    def _build_layer(self, name, inputs, k, use_in=True, use_dropout=False):
        layer = dict()
        with tf.variable_scope(name):
            layer['filters'] = tf.get_variable('filters', [4, 4, get_shape(inputs)[-1], k])
            layer['conv'] = tf.nn.conv2d(inputs, layer['filters'], strides=[1, 2, 2, 1], padding='SAME')
            layer['bn'] = inst_norm(layer['conv']) if use_in else layer['conv']
            layer['fmap'] = lkrelu(layer['bn'], slope=0.2)
        return layer

    def _build_discriminator(self, inputs, reuse=None):
        discriminator = dict()

        # C64-C128-C256-C512 -> PatchGAN
        discriminator['l1'] = self._build_layer('l1', inputs, 64, use_in=False)
        discriminator['l2'] = self._build_layer('l2', discriminator['l1']['fmap'], 128)
        discriminator['l3'] = self._build_layer('l3', discriminator['l2']['fmap'], 256)
        discriminator['l4'] = self._build_layer('l4', discriminator['l3']['fmap'], 512)
        with tf.variable_scope('l5'):
            l5 = dict()
            l5['filters'] = tf.get_variable('filters', [4, 4, get_shape(discriminator['l4']['fmap'])[-1], 1])
            l5['conv'] = tf.nn.conv2d(discriminator['l4']['fmap'], l5['filters'], strides=[1, 1, 1, 1], padding='SAME')
            l5['fmap'] = l5['conv'] # no sigmoid because we use LSGAN loss
            discriminator['l5'] = l5
        return discriminator
