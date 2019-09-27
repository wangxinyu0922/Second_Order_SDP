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

import numpy as np
import tensorflow as tf

from .base_bucket import BaseBucket

import pdb
#***************************************************************
class DictBucket(BaseBucket):
  """"""
  
  #=============================================================
  def __init__(self, idx, depth, config=None):
    """"""
    
    super(DictBucket, self).__init__(idx, config=config)
    
    self._depth = depth
    self._indices = []
    self._tokens = []
    self._str2idx = {}
    
    return
  
  #=============================================================
  def reset(self):
    """"""
    
    self._indices = []
    self._tokens = []
    self._str2idx = {}
    return

  #=============================================================
  def add(self, indices, tokens):
    """"""
    
    assert self._is_open, 'DictBucket is not open for adding entries'
    
    string = ' '.join(tokens)
    if string in self._str2idx:
      sequence_index = self._str2idx[string]
    else:
      sequence_index = len(self._indices)
      self._str2idx[string] = sequence_index
      self._tokens.append(tokens)
      super(DictBucket, self).add(indices)
    return sequence_index
  
  #=============================================================
  def close(self):
    """"""
    # Initialize the index matrix
    first_dim = len(self._indices)
    second_dim = max(len(indices) for indices in self._indices) if self._indices else 0
    shape = [first_dim, second_dim]
    if self.depth > 0:
      shape.append(self.depth)
    elif self.depth == -1:
      shape.append(shape[-1])
    

    attr_mode=False
    if self._indices:
      # if type(self._indices[0][0])==type((0,0)):
      #   pdb.set_trace()
      if type(self._indices[0][0])==type(np.array(0)):
        shape.append(self._indices[0][0].shape[0])
        attr_mode=True

    data = np.zeros(shape, dtype=np.int32)
    # Add data to the index matrix
    if self.depth >= 0:
      try:
        for i, sequence in enumerate(self._indices):
          if sequence:
            if attr_mode:
              sequence=np.array(sequence)
            data[i, 0:len(sequence)] = sequence
            
      except ValueError:
        print('Expected shape: {}\nsequence: {}'.format([len(sequence), self.depth], sequence))
        print('\ntokens: {}'.format(self._tokens[i]))
        raise
    elif self.depth == -1:
      # for graphs, sequence should be list of (idx, val) pairs
      for i, sequence in enumerate(self._indices):
        for j, node in enumerate(sequence):
          for edge in node:
            if isinstance(edge, (tuple, list)):
              edge, v = edge
              data[i, j, edge] = v
            else:
              data[i, j, edge] = 1
    super(DictBucket, self).close(data)
    
    return
  #=============================================================
  def bert_close(self,sep_token=None,is_pretrained=False,get_dephead=False):
    # Data Preprocess specially for bert
    if is_pretrained:
      first_dim = len(self._indices)
      second_dim = max(indices.shape[0] for indices in self._indices) if self._indices else 0
      third_dim = self._indices[0].shape[-1] if self._indices else 0
      shape = [first_dim, second_dim, third_dim]
      data = np.zeros(shape, dtype=np.float32)
      if get_dephead:
        assert len(self._tokens)==len(self._indices), "inconsistant of tokens and features!"
      for i,indices in enumerate(self._indices):
        if get_dephead:
          tokens=self._tokens[i]
          assert len(tokens)==indices.shape[0], "inconsistant of tokens and features!"
          data[i,:indices.shape[0]]=indices[[int(x) for x in tokens]]
        else:
          data[i,:indices.shape[0]]=indices
      super(DictBucket, self).close(data)
      return
    first_dim = len(self._indices)
    
    # if first_dim>0:
    #   pdb.set_trace()
    
    bertlist=[]
    bertmask=[]
    token_mapping=[]
    for orig_tokens in self._indices:
      bert_tokens=[]
      orig_to_tok_map=[]
      bert_tokens.append(sep_token[0])
      for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(orig_token)
      bert_tokens.append(sep_token[1])

      bertlist.append(bert_tokens)
      bertmask.append([1]*len(bert_tokens))
      token_mapping.append(orig_to_tok_map)
    # Initialize the index matrix
    bert_tokens_dim = max(len(indices) for indices in bertlist) if bertlist else 0
    bertmask_dim = max(len(indices) for indices in bertmask) if bertmask else 0
    mapping_dim = max(len(indices) for indices in token_mapping) if token_mapping else 0
    
    bert_tokens_shape = [first_dim, bert_tokens_dim]
    bertmask_shape = [first_dim, bertmask_dim]
    mapping_shape = [first_dim, mapping_dim]

    bert_tokens_data = np.zeros(bert_tokens_shape, dtype=np.int32)
    segment_data = np.zeros(bert_tokens_shape, dtype=np.int32)
    bertmask_data = np.zeros(bertmask_shape, dtype=np.int32)
    mapping_data = np.zeros(mapping_shape, dtype=np.int32)
    for i in range(len(bertlist)):
      if bertlist[i]:
        #bert token should have the same shape as bert mask
        bert_tokens_data[i, 0:len(bertlist[i])] = bertlist[i]
        bertmask_data[i, 0:len(bertmask[i])] = bertmask[i]
      if token_mapping[i]:
        mapping_data[i, 0:len(token_mapping[i])] = token_mapping[i]

    #set the bert dictionary
    data={}
    data['input_ids']=bert_tokens_data
    data['input_mask']=bertmask_data
    data['segment_ids']=segment_data
    data['mapping']=mapping_data
    super(DictBucket, self).close(data)
    return
  #=============================================================
  def elmo_close(self):
    # Data Preprocess specially for bert or elmo
    first_dim = len(self._indices)
    second_dim = max(indices.shape[0] for indices in self._indices) if self._indices else 0
    third_dim = self._indices[0].shape[-1] if self._indices else 0
    shape = [first_dim, second_dim, 3, third_dim]
    data = np.zeros(shape, dtype=np.float32)
    for i,indices in enumerate(self._indices):
      data[i,:indices.shape[0]]=indices
    super(DictBucket, self).close(data)
    return
  #=============================================================
  @property
  def depth(self):
    return self._depth
  @property
  def data_indices(self):
    return self._data
