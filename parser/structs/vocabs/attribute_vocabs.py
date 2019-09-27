#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2017 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import os
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.structs.vocabs.base_vocabs import CountVocab
from parser.structs.vocabs.token_vocabs import GraphTokenVocab
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, embeddings, classifiers
import pdb

class AttributeVocab(GraphTokenVocab):
  def __init__(self, *args, **kwargs):
    """"""
    super(AttributeVocab, self).__init__(*args, **kwargs)
    
    return
    #=============================================================
  def get_bilinear_classifier(self, layer, outputs, token_weights, variable_scope=None, reuse=False, debug=False):
    """"""
    
    recur_layer = layer
    hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
    add_linear = self.add_linear
    with tf.variable_scope(variable_scope or self.field):
      for i in six.moves.range(0, self.n_layers-1):
        with tf.variable_scope('FC-%d' % i):
          layer = classifiers.hidden(layer, 2*self.hidden_size,
                                      hidden_func=self.hidden_func,
                                      hidden_keep_prob=hidden_keep_prob)
      with tf.variable_scope('FC-top'):
        layers = classifiers.hiddens(layer, 2*[self.hidden_size],
                                    hidden_func=self.hidden_func,
                                    hidden_keep_prob=hidden_keep_prob)
      layer1, layer2 = layers.pop(0), layers.pop(0)
      
      with tf.variable_scope('Classifier'):
        if self.diagonal:
          logits = classifiers.diagonal_bilinear_classifier(
            layer1, layer2, len(self),
            hidden_keep_prob=hidden_keep_prob,
            add_linear=add_linear)
        else:
          logits = classifiers.bilinear_classifier(
            layer1, layer2, len(self),
            hidden_keep_prob=hidden_keep_prob,
            add_linear=add_linear)
    
    #-----------------------------------------------------------
    # Process the targets
    # (n x m x m)
    label_targets = self.placeholder
    unlabeled_predictions = outputs['unlabeled_predictions']
    unlabeled_targets = outputs['unlabeled_targets']
    
    #-----------------------------------------------------------
    # Process the logits
    # (n x m x c x m) -> (n x m x m x c)
    transposed_logits = tf.transpose(logits, [0,1,3,2])
    
    #-----------------------------------------------------------
    # Compute the probabilities/cross entropy
    # (n x m x m) -> (n x m x m x 1)
    head_probabilities = tf.expand_dims(tf.stop_gradient(outputs['probabilities']), axis=-1)
    # (n x m x m x c) -> (n x m x m x c)
    label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
    # (n x m x m), (n x m x m x c), (n x m x m) -> ()
    label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, transposed_logits, weights=token_weights*unlabeled_targets)
    
    #-----------------------------------------------------------
    # Compute the predictions/accuracy
    # (n x m x m x c) -> (n x m x m)
    #print('23333')
    predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32)
    # (n x m x m) (*) (n x m x m) -> (n x m x m)
    true_positives = nn.equal(label_targets, predictions) * unlabeled_predictions
    correct_label_tokens = nn.equal(label_targets, predictions) * unlabeled_targets
    # (n x m x m) -> ()
    n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
    n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
    n_true_positives = tf.reduce_sum(true_positives)
    n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
    # () - () -> ()
    n_false_positives = n_unlabeled_predictions - n_true_positives
    n_false_negatives = n_unlabeled_targets - n_true_positives
    # (n x m x m) -> (n)
    n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
    n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
    n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1,2])
    # (n) x 2 -> ()
    n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
    n_correct_label_sequences = tf.reduce_sum(nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))
    
    #-----------------------------------------------------------
    # Populate the output dictionary
    rho = self.loss_interpolation
    outputs['label_predictions']=predictions
    outputs['label_targets'] = label_targets
    outputs['probabilities'] = label_probabilities * head_probabilities
    outputs['label_loss'] = label_loss
    # Combination of labeled loss and unlabeled loss
    outputs['loss'] = 2*((1-rho) * outputs['loss'] + rho * label_loss)
    
    outputs['n_true_positives'] = n_true_positives
    outputs['n_false_positives'] = n_false_positives
    outputs['n_false_negatives'] = n_false_negatives
    outputs['n_correct_sequences'] = n_correct_sequences
    outputs['n_correct_label_tokens'] = n_correct_label_tokens
    outputs['n_correct_label_sequences'] = n_correct_label_sequences
    return outputs

