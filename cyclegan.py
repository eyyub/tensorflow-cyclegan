import numpy as np
import tensorflow as tf
from model.discriminator import Discriminator
from model.generator import Generator

class _ImagePool(object):
    def __init__(self, max_size):
        self._pool = []
        self._max_size = max_size

    def sample_random(self, a):
        if len(self._pool) < self._max_size:
            self._pool.append(a)
            return a
        r = np.random.random()
        if r > 0.5:
            r = np.random.randint(0, len(self._pool) - 1)
            inst = self._pool[r]
            self._pool[r] = a
            return inst
        else:
            return a

class CycleGAN(object):
    def __init__(self, width, height, xchan, ychan, lambda_=10., pool_size=50, lr=0.0002, beta1=0.5):
        """
            width: image width in pixel.
            height: image height in pixel.
            ichan: number of channels used by input images.
            ochan: number of channels used by output images.
            lambda_: Cycle-Consistency weighting.
            pool_size: Image pool size.
            lr: learning rate for ADAM optimizer.
            beta1: beta1 parameter for ADAM optimizer.
        """

        self._dx_pool = _ImagePool(pool_size)
        self._dy_pool = _ImagePool(pool_size)

        self._xs = tf.placeholder(tf.float32, [None, width, height, xchan])
        self._ys = tf.placeholder(tf.float32, [None, width, height, ychan])

        self._d_xs = tf.placeholder(tf.float32, [None, width, height, xchan])
        self._d_ys = tf.placeholder(tf.float32, [None, width, height, ychan])
        self._fake_d_xs = tf.placeholder(tf.float32, [None, width, height, xchan])
        self._fake_d_ys = tf.placeholder(tf.float32, [None, width, height, ychan])

        self._gx = Generator('Gx', self._ys, xchan)
        self._gy = Generator('Gy', self._xs, ychan)

        self._gx_from_gy = Generator('Gx', self._gy['l15']['fmap'], xchan, reuse=True)
        self._gy_from_gx = Generator('Gy', self._gx['l15']['fmap'], ychan, reuse=True)

        self._real_dx = Discriminator('Dx', self._d_xs)
        self._fake_dx = Discriminator('Dx', self._xs, reuse=True)
        self._fake_dx_g = Discriminator('Dx', self._gx['l15']['fmap'], reuse=True)

        self._real_dy = Discriminator('Dy', self._d_ys)
        self._fake_dy = Discriminator('Dy', self._ys, reuse=True)
        self._fake_dy_g = Discriminator('Dy', self._gy['l15']['fmap'], reuse=True)

        # Forward and backward Cycle-Consistency with LSGAN-kind losses
        cycle_loss = lambda_ * (tf.reduce_mean(tf.abs((self._gx_from_gy['l15']['fmap'] - self._xs))) + tf.reduce_mean(tf.abs((self._gy_from_gx['l15']['fmap'] - self._ys))))
        self._gx_loss =  0.5 * tf.reduce_mean(tf.square(self._fake_dx_g['l5']['fmap'] - 1.)) + cycle_loss
        self._gy_loss =  0.5 * tf.reduce_mean(tf.square(self._fake_dy_g['l5']['fmap'] - 1.)) + cycle_loss

        self._dx_loss =  0.5 * tf.reduce_mean(tf.square(self._real_dx['l5']['fmap'] - 1.)) + 0.5 * tf.reduce_mean(tf.square(self._fake_dx['l5']['fmap']))
        self._dy_loss =  0.5 * tf.reduce_mean(tf.square(self._real_dy['l5']['fmap'] - 1.)) + 0.5 * tf.reduce_mean(tf.square(self._fake_dy['l5']['fmap']))

        self._gx_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._gx_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gx'))

        self._gy_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._gy_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gy'))
    
        self._dx_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._dx_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dx'))

        self._dy_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._dy_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dy'))

    def train_step(self, sess, xs, ys, d_xs, d_ys):

        ops = [self._gx_train_step, self._gy_train_step, self._gx_loss, self._gy_loss, self._gx['l15']['fmap'], self._gy['l15']['fmap']]
        _, _, gxloss_curr, gyloss_curr, gxs, gys = sess.run(ops, feed_dict={self._xs: xs, self._ys: ys})

        _, _, dxloss_curr, dyloss_curr = sess.run([self._dx_train_step, self._dy_train_step, self._dx_loss, self._dy_loss],
            feed_dict={self._xs: self._dx_pool.sample_random(gxs),
                       self._ys: self._dy_pool.sample_random(gys),
                       self._d_xs: d_xs, self._d_ys: d_ys})

        return ((gxloss_curr, gyloss_curr), (dxloss_curr, dyloss_curr))

    def sample_gx(self, sess, ys):
        return sess.run(self._gx['l15']['fmap'], feed_dict={self._ys: ys})

    def sample_gy(self, sess, xs):
        return sess.run(self._gy['l15']['fmap'], feed_dict={self._xs: xs})
