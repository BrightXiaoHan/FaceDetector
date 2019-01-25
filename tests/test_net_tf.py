import unittest

import numpy as np
import tensorflow as tf

import mtcnn.network.mtcnn_tensorflow as mtcnn_tf
import mtcnn.utils.functional as func


class TestMtcnntf(unittest.TestCase):

    def test_pnet(self):
        # data = np.random.randn(128, 12, 12)
        # labels = np.random.randint(0, 2, (128,))
        # boxreg = np.random.randn(128, 4)
        # landmarks = np.random.randn(128, 10)

        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        labels = tf.placeholder(tf.int8, (None, 3), 'cls')
        boxreg = tf.placeholder(tf.float32, (None, 4), 'breg')
        landmarks = tf.placeholder(tf.float32, (None, 10), 'landm')

        with tf.Session() as sess:

            pnet = mtcnn_tf.PNet(
                {'data': data, "labels": labels, 'boxreg': boxreg, "landmarks": landmarks})
            

