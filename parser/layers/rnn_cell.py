# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from thumt.layers.nn import linear

import pdb
from parser.neural import nn

class LegacyGRUCell(tf.nn.rnn_cell.RNNCell):
    """ Groundhog's implementation of GRUCell

    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None):
        super(LegacyGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            # pdb.set_trace()
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            all_inputs = list(inputs) + [state]
            r = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="reset_gate"))
            u = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="update_gate"))
            all_inputs = list(inputs) + [r * state]
            c = linear(all_inputs, self._num_units, True, False,
                       scope="candidate")
            # pdb.set_trace()
            new_state = (1.0 - u) * state + u * tf.tanh(c)

        return new_state, new_state


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


class LegacyLSTMCell(tf.nn.rnn_cell.RNNCell):
    """ Groundhog's implementation of GRUCell

    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None):
        super(LegacyLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units


    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope, default_name="lstm_cell",
                               values=[inputs, state]):

            i = tf.nn.sigmoid(linear(inputs, self._num_units, False, False,
                                     scope="input_gate"))
            f = tf.nn.sigmoid(linear(inputs, self._num_units, False, False,
                                     scope="forget_gate"))
            o = tf.nn.sigmoid(linear(inputs, self._num_units, False, False,
                                     scope="outut_gate"))
            c_ = tf.nn.sigmoid(linear(inputs, self._num_units, False, False,
                                     scope="cell_gate"))

            new_state = f * state + i * c_

            new_inputs = o * tf.tanh(new_state)

        return new_inputs, new_state


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units