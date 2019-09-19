#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from parser_model.neural.optimizers.optimizer import Optimizer


# ***************************************************************
class BERTAdamOptimizer(Optimizer):
    """"""
    def __init__(self, config=None):
        super(BERTAdamOptimizer, self).__init__(config=config)


    # =============================================================
    def dense_update(self, gradient, variable):
        """"""

        updates = []

        if self.mu > 0:
            mean_update = self.dense_moving_average(
                variable, gradient,
                name='Mean',
                decay=self.mu)
            mean, _ = mean_update
            mean = (1 - self.gamma) * mean + self.gamma * gradient
            updates.extend(mean_update)
        else:
            mean = gradient

        if self.nu > 0:
            zero_deviation_update = self.dense_moving_average(
                variable, gradient ** 2,
                name='ZeroDeviation',
                decay=self.nu)
            zero_deviation, _ = zero_deviation_update
            zero_deviation = tf.sqrt(zero_deviation + self.epsilon)
            updates.extend(zero_deviation_update)
        else:
            zero_deviation = 1

        variable_step = self.annealed_learning_rate * mean / zero_deviation
        variable_step = tf.where(tf.is_finite(variable_step),
                                 variable_step,
                                 tf.zeros_like(variable_step))
        return variable_step, updates

    @property
    def num_train_steps(self):
        return self._config.getint(self, 'num_train_steps')
    @property
    def num_warmup_steps(self):
        return self._config.getint(self, 'num_warmup_steps')
    @property
    def num_warmup_steps(self):
        return self._config.getint(self, 'num_warmup_steps')
    @property
    def weight_decay_rate(self):
        return self._config.getfloat(self, 'weight_decay_rate')
    @property
    def beta_1(self):
        return self._config.getfloat(self, 'beta_1')
    @property
    def beta_2(self):
        return self._config.getfloat(self, 'beta_2')
    @property
    def epsilon(self):
        return self._config.getfloat(self, 'epsilon')
    @property
    def exclude_from_weight_decay(self):
        return self._config.getstr(self, 'exclude_from_weight_decay')
