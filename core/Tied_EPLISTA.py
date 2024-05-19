#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Tied version of EPLISTA.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_ss, shrink_free
from models.LISTA_base import LISTA_base


class Tied_EPLISTA(LISTA_base):
    """
    Implementation of LISTA model proposed by LeCun in 2010.
    """

    def __init__(self, A, T, lam, percent, max_percent, coord, scope):
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

        self._p = percent
        self._maxp = max_percent
        self._scale = 1.001 * np.linalg.norm(A, ord=2) ** 2

        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta

        self._ps = [(t + 1) * self._p for t in range(self._T)]
        self._ps = np.clip(self._ps, 0.0, self._maxp)

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
        alphas_ = []  # step sizes
        gammas_ = []
        W = (np.transpose(self._A) / self._scale).astype(np.float32)
        Q = np.eye(self._N).astype(np.float32)

        with tf.variable_scope(self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant(value=self._A, dtype=tf.float32)

            Ws_.append(tf.get_variable(name='W', dtype=tf.float32, initializer=W))
            Qs_.append(tf.get_variable(name='Q', dtype=tf.float32, initializer=Q))

            Ws_ = Ws_ * self._T
            Qs_ = Qs_ * self._T
            for t in range(self._T):
                thetas_.append(tf.get_variable(name="theta_%d" % (t + 1), dtype=tf.float32, initializer=self._theta))
                alphas_.append(tf.get_variable(name="alpha_%d" % (t + 1), dtype=tf.float32, initializer=1.0))
                gammas_.append(tf.get_variable(name="gamma_%d" % (t + 1), dtype=tf.float32, initializer=0.0))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list(zip(Ws_, Qs_, thetas_, alphas_, gammas_))

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
                W_, Q_, theta_, alpha_, gamma_ = self.vars_in_layer[t]
                percent = self._ps[t]

                res_ = y_ - tf.matmul(self._kA_, xh_)
                zh_ = xh_ + alpha_ * tf.matmul(W_, res_) + gamma_ * tf.matmul(Q_, xh_)
                xh_ = shrink_ss(zh_, theta_, percent)
                xhs_.append(xh_)

        return xhs_
