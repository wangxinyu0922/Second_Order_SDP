#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2019 Xinyu Wang
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
import pdb
import os
import codecs
import zipfile
import gzip
try:
  import lzma
except:
  try:
    from backports import lzma
  except:
    import warnings
    warnings.warn('Install backports.lzma for xz support')
try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl
from collections import Counter

import numpy as np
import tensorflow as tf
 
from parser.structs.vocabs.base_vocabs import SetVocab
from . import conllu_vocabs as cv
from parser.neural import embeddings,nonlin,classifiers

class ElmoVocab(SetVocab, cv.FormVocab):
  """"""
  
  #=============================================================
  def __init__(self, Bert_file=None, name=None, config=None):
    """"""
    if (Bert_file is None) != (name is None):
      raise ValueError("You can't pass in a value for only one of Bert_file and name to BertVocab.__init__")
    super(ElmoVocab, self).__init__(config=config)
    self.placeholder = tf.placeholder(tf.float32, [None,None,3,1024], name=self.classname+'_feats')
    
    return
  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""
    #pdb.set_trace()
    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    #pdb.set_trace()
    with tf.variable_scope('elmo_vocab'):
      if self.strategy=='three_layers':
        weight = tf.get_variable('softmax_weight', shape=[3], initializer=tf.ones_initializer)
        softmax_weight = tf.nn.softmax(weight)
        layer=tf.einsum('nabc,b->nac',self.placeholder,weight)
      elif self.strategy=='two_layers':
        weight = tf.get_variable('weight', initializer=0.5)
        layer = self.placeholder[:,:,-1]*weight+(1-weight)*self.placeholder[:,:,-2]
      elif self.strategy=='third_layer':
        layer=self.placeholder[:,:,-1,:]
      elif self.strategy=='second_layer':
        layer=self.placeholder[:,:,-2,:]
      else:
        assert 0, 'please specify the strategy'
      scalar = tf.get_variable('scalar', shape=[1], initializer=tf.ones_initializer)
      layer=scalar*layer
      layer=classifiers.hidden(layer,self.linear_size,hidden_func=self.hidden_func)
    return layer
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    #pdb.set_trace()
    feed_dict[self.placeholder] = indices
    return feed_dict
  def add(self, token):
    tokens=self.tokenizer.tokenize(token)
    token_ids=self.tokenizer.convert_tokens_to_ids(tokens)

    return token_ids
  #=============================================================
  def count(self, *args):
    """"""
    return True

  #=============================================================
  def dump(self):
    if self.save_as_pickle and not os.path.exists(self.vocab_loadname):
      os.makedirs(os.path.dirname(self.vocab_loadname), exist_ok=True)
      with open(self.vocab_loadname, 'wb') as f:
        pkl.dump((self._tokens, self._embeddings), f, protocol=pkl.HIGHEST_PROTOCOL)
    return

  #=============================================================
  def load(self):
    """"""
    return True
  def get_elmo_path(self,dataset):
    return os.path.join(self._config.getstr(self, 'elmo_path'),self._config.getstr(self,'elmo_tag')+dataset+'.hdf5')
  #=============================================================
  @property
  def Bert_file(self):
    return self._config.getstr(self, 'Bert_file')
  @property
  def vocab_loadname(self):
    return self._config.getstr(self, 'vocab_loadname')
  @property
  def name(self):
    return self._name
  @property
  def max_embed_count(self):
    return self._config.getint(self, 'max_embed_count')
  @property
  def embeddings(self):
    return self._embeddings
  @property
  def embed_keep_prob(self):
    return self._config.getfloat(self, 'max_embed_count')
  @property
  def embed_size(self):
    return self._embed_size
  @property
  def save_as_pickle(self):
    return self._config.getboolean(self, 'save_as_pickle')
  @property
  def linear_size(self):
    return self._config.getint(self, 'linear_size')
  @property
  def use_client(self):
    return self._config.getboolean(self, 'use_client')
  @property
  def is_training(self):
    return self._config.getboolean(self, 'is_training')
  @property
  def get_elmo_path(self):
    return os.path.join(self._config.getstr(self, 'elmo_path'),self._config.getstr(self,'elmo_tag')+'.hdf5')
    
  @property
  def strategy(self):
    return self._config.getstr(self, 'strategy')
  @property
  def hidden_func(self):
    hidden_func = self._config.getstr(self, 'hidden_func')
    if hasattr(nonlin, hidden_func):
      return getattr(nonlin, hidden_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
