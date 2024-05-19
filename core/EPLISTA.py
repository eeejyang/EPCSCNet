#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Learned ISTA with Elastic Projection.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_free
from models.LISTA_base import LISTA_base


class EPLISTA(LISTA_base):
    """
    Implementation of LISTA model proposed by LeCun in 2010.
    """

    def __init__(self, A, T, lam, coord, scope):
        """
        :A      : Numpy ndarray. Dictionary/Sensing matrix.
        :T      : Integer. Number of layers (depth) of this LISTA model.
        :lam    : Float. The initial weight of l1 loss term in LASSO.
        :untied : Boolean. Flag of whether weights are shared within layers.
        :scope  : String. Scope name of the model.
        """
        self._A = A.astype(np.float32)
        self._T = T
        self._lam = lam
        self._M = self._A.shape[0]
        self._N = self._A.shape[1]

        self._scale = 1.001 * np.linalg.norm(A, ord=2) ** 2

        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta

        self._coord = coord
        self._scope = scope

        """ Set up layers."""
        self.setup_layers()

    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """

        Ws_ = []
        Qs_ = []
        thetas_ = []
        W = (np.transpose(self._A) / self._scale).astype(np.float32)
        Q = np.zeros((self._N, self._N), dtype=np.float32)
        scale = [0 for i in range(self._T)]

        with tf.variable_scope(self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant(value=self._A, dtype=tf.float32)

            Ws_.append(tf.get_variable(name='W', dtype=tf.float32, initializer=W))

            Ws_ = Ws_ * self._T

            for t in range(self._T):
                thetas_.append(tf.get_variable(name="theta_%d" % (t + 1), dtype=tf.float32, initializer=self._theta))
                Qs_.append(tf.get_variable(name='Q_%d' % (t + 1), dtype=tf.float32, initializer=Q * scale[t]))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list(zip(Ws_, Qs_, thetas_))

    def inference(self, y_, x0_=None):
        xhs_ = []  # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = tf.shape(y_)[-1]
            xh_ = tf.zeros(shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_

        xhs_.append(xh_)

        with tf.variable_scope(self._scope, reuse=True) as vs:
            for t in range(self._T):
                W_, Q_, theta_ = self.vars_in_layer[t]

                res_ = y_ - tf.matmul(self._kA_, xh_)
                zh_ = xh_ + (tf.matmul(W_, res_) + tf.matmul(Q_, xh_))
                xh_ = shrink_free(zh_, theta_)
                xhs_.append(xh_)

        return xhs_
