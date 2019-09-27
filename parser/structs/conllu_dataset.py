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
import re
import codecs
import zipfile
import gzip
try:
	import lzma
except ImportError:
	try:
		from backports import lzma
	except ImportError:
		import warnings
		warnings.warn('Install backports.lzma for xz support')
from collections import Counter

import numpy as np
import tensorflow as tf
import pdb
import math
from parser.structs.buckets import DictMultibucket
import json

import h5py as h5

#***************************************************************
class CoNLLUDataset(set):
	""""""
	
	#=============================================================
	def __init__(self, conllu_files, vocabs, config=None):
		""""""
		super(CoNLLUDataset, self).__init__(vocabs)
		self._multibucket = DictMultibucket(vocabs, max_buckets=config.getint(self, 'max_buckets'), config=config)
		self._is_open = False
		self._config = config
		self._conllu_files = conllu_files
		assert len(conllu_files) > 0, "You didn't pass in any valid CoNLLU files! Maybe you got the path wrong?"
		self._cur_file_idx = -1
		self.id_buff=[]
		self.use_elmo=False
		self.use_bert=False
		self.use_seq2seq=False
		self.seq2seq_data=None
		#---------------------------------------------------------
		self.seq2seq_node_indices = {}
		self.seq2seq_node_labels = {}
		#---------------------------------------------------------
		for vocab in self:
			if 'Elmo' in vocab.classname:
				self.use_elmo=True
				self.Elmo_data=h5.File(vocab.get_elmo_path,'r')
			if 'Bert' in vocab.classname:
				if vocab.get_pretrained==True:
					self.use_bert=True
					self.Bert_data=h5.File(vocab.get_pretrained_bert_path,'r')
			if 'seq2seq' in vocab.classname.lower():
				self.use_seq2seq=True
				if self.seq2seq_data==None and hasattr(vocab, 'get_nodes_path'):
					self.seq2seq_data=json.load(open(vocab.get_nodes_path,'r'))
		self.load_next()
		return
	
	#=============================================================
	def reset(self):
		""""""
		
		self._multibucket.reset(self)
		for vocab in self:
			vocab.reset()
		return
		
	#=============================================================
	def load_next(self, file_idx=None):
		""""""
		
		if self._cur_file_idx == -1 or len(self.conllu_files) > 1:
			self.reset()
		
			if file_idx is None:
				self._cur_file_idx = (self._cur_file_idx + 1) % len(self.conllu_files)
				file_idx = self._cur_file_idx
			
			with self.open():
				for index, sent in enumerate(self.itersents(self,self.conllu_files[file_idx])):
					#pdb.set_trace()
					self.add(sent, index)
		return
	
	#=============================================================
	def open(self):
		""""""
		
		self._multibucket.open()
		for vocab in self:
			vocab.open()
		
		self._is_open = True
		return self
		
	#=============================================================
	def add(self, sent, sent_idx=None):
		""""""
		self.seq2seq_node_indices[sent_idx] = {}
		self.seq2seq_node_labels[sent_idx] = {}
		assert self._is_open, 'The CoNLLUDataset is not open for adding entries'
		# pdb.set_trace()
		sent_tokens = {}
		sent_indices = {} 
		set_bucket_size=None
		for vocab in self:
			#vocab[1].classname to set the vocab inputs
			#sent_tokens[vocab.classname]=[1]*sent_tokens[vocab.classname].shape[0]
			#pdb.set_trace()
			if not 'seq2seq' in vocab.classname.lower():
				if hasattr(vocab,'conllu_idx'):
					try:
						tokens = [line[vocab.conllu_idx] for line in sent]
					except:
						pdb.set_trace()

					if vocab.classname=='DepheadBertVocab':
						tokens.insert(0,'0')
					else:
						tokens.insert(0, vocab.get_root())

				if self.use_elmo and 'Elmo' in vocab.classname:
					if sent!=[]:
						#pdb.set_trace()
						indices=self.Elmo_data[self.id_buff[-1]][...]
						assert indices.shape[0]-1==len(sent), 'inconsistent of data size and sentence length!'
					else:
						indices=[]
					#pdb.set_trace()
				elif self.use_bert and 'Bert' in vocab.classname:
					if sent!=[]:
						#pdb.set_trace()
						indices=self.Bert_data[self.id_buff[-1]][...]
						try:
							assert indices.shape[0]-1==len(sent), 'inconsistent of data size and sentence length!'
						except:
							pdb.set_trace()
					else:
						indices=[]
				else:
					try:
						indices = vocab.add_sequence(tokens) # for graphs, list of (head, label) pairs
					except:
						pdb.set_trace()
				sent_tokens[vocab.classname] = tokens
				sent_indices[vocab.classname] = indices
		#pdb.set_trace()
		self._multibucket.add(sent_indices, sent_tokens, length=len(sent)+1)
		#pdb.set_trace()
		# ------------------------------------------------------------------------------------
		for vocab in self:
			if self.use_seq2seq and 'seq2seq' in vocab.classname.lower():
				current_data = self.seq2seq_data[self.id_buff[-1][1:]]
				# tokens = [current_data[data][vocab.mrp_idx] if current_data[data][vocab.mrp_idx] != '' else 0 for data in current_data]
				#pdb.set_trace()
				if vocab.mrp_idx in current_data['nodes'][0]:
					tokens = [data[vocab.mrp_idx] if data[vocab.mrp_idx] != '' else 0 for data in current_data['nodes']]
					tokens.insert(0, vocab.get_root())
					if vocab.field not in ['semhead', 'semrel']:
						tokens.insert(0, vocab.get_bos())
						tokens.append(vocab.get_eos())
				else:
					tokens = current_data[vocab.mrp_idx]
				# pdb.set_trace()

				indices = vocab.add_sequence(tokens)
				self.seq2seq_node_indices[sent_idx][vocab.classname] = indices
				self.seq2seq_node_labels[sent_idx][vocab.classname] = tokens
		#--------------------------------------------------------------------------------------
		return



	#=============================================================
	def close(self):
		""""""
		
		self._multibucket.close()
		for vocab in self:
			vocab.close()
		self._is_open = False
		return 
	
	#=============================================================
	def batch_iterator(self, shuffle=False):
		""""""
		
		assert self.batch_size > 0, 'batch_size must be > 0'
		#pdb.set_trace()
		batches = []
		bucket_indices = self._multibucket.bucket_indices
		for i in np.unique(bucket_indices):
			bucket_size = self._multibucket.max_lengths[i] 
			subdata = np.where(bucket_indices == i)[0]
			if len(subdata) > 0:
				if shuffle:
					np.random.shuffle(subdata)
				#so if here is 1.99, it will still be one split?
				n_splits = math.ceil(subdata.shape[0] * bucket_size / self.batch_size)
				#n_splits = max(subdata.shape[0] * bucket_size // self.batch_size, 1)
				splits = np.array_split(subdata, n_splits)
				batches.extend(splits)
		if shuffle:
			np.random.shuffle(batches)
		return iter(batches)
		
	#=============================================================
	def set_placeholders(self, indices, feed_dict={}):
		""""""
		sent_length = 0
		for vocab in self:
			if vocab.classname == 'XPOSTokenVocab':
				data = self._multibucket.get_data(vocab.classname, indices)
				sent_length = data.shape[-1]
				break

		for vocab in self:
			# pdb.set_trace()
			# if vocab == 'LabelTokenVocab':
			# 	data = self._multibucket.get_data(vocab.classname, indices)
			# 	print('******************************************')
			# 	print(data)
			# 	print('******************************************')
			#---------------------------------------------------------------------
			if self.use_seq2seq and 'seq2seq' in vocab.classname.lower():
				node = []
				#pdb.set_trace()
				for index in indices:
					node.append(self.seq2seq_node_indices[index][vocab.classname])
				data = self.node_padding(node, vocab.depth, vocab.classname, sent_length)
				feed_dict = vocab.set_placeholders(data, feed_dict=feed_dict)
			else:
				#--------------------------------------------------------------------------
				data = self._multibucket.get_data(vocab.classname, indices)
				feed_dict = vocab.set_placeholders(data, feed_dict=feed_dict)
			# pdb.set_trace()

		return feed_dict


	#-------------------------------------------------------------------------------
	def node_padding(self, node, depth, vocab, sent_length):
		# Padding according nodes of sentences of one batch
		# node: if 2 dims, list [[1,2],[1,2,3]] -> node_padding: list [[1,2,0],[1,2,3]]

		# Initialize the index matrix
		first_dim = len(node)
		second_dim = max(len(indices) for indices in node) if node else 0
		shape = [first_dim, second_dim]
		if depth > 0:
			shape.append(depth)
		elif depth == -1:
			shape.append(shape[-1])
		elif depth == -2:
			shape.append(shape[-1])
		attr_mode = False
		if node:
			# if type(self._indices[0][0])==type((0,0)):
			#   pdb.set_trace()
			if type(node[0][0]) == type(np.array(0)):
				shape.append(node[0][0].shape[0])
				attr_mode = True

		data = np.zeros(shape, dtype=np.int32)
		# Add data to the index matrix
		if depth >= 0:
			try:
				for i, sequence in enumerate(node):
					if sequence:
						if attr_mode:
							sequence = np.array(sequence)
						try:
							data[i, 0:len(sequence)] = sequence
						except:
							pdb.set_trace()

			except ValueError:
				print('wrong...............')
				raise
		elif depth == -1:
			# for graphs, sequence should be list of (idx, val) pairs
			for i, sequence in enumerate(node):
				for j, node in enumerate(sequence):
					for edge in node:
						if isinstance(edge, (tuple, list)):
							edge, v = edge
							data[i, j, edge] = v
						else:
							data[i, j, edge] = 1
		elif depth == -2 and not vocab == 'Seq2SeqSrcCopyMapVocab':
			for i, mapping in enumerate(node):
				if mapping:
					# pdb.set_trace()
					data[i,0:len(mapping),0:len(mapping)]=np.array(mapping)
		elif depth == -2 and vocab == 'Seq2SeqSrcCopyMapVocab':
			shape = [first_dim, sent_length, sent_length]
			data = np.zeros(shape, dtype=np.int32)
			for i, mapping in enumerate(node):
				if mapping:
					# pdb.set_trace()
					data[i, 0:len(mapping), 0:len(mapping)] = np.array(mapping)


		return data
	#--------------------------------------------------------------------------

	#=============================================================
	def get_tokens(self, indices):
		""""""
		token_dict = {}
		for vocab in self:
			token_dict[vocab.field] = self._multibucket.get_tokens(vocab.classname, indices)
		lengths = self._multibucket.lengths[indices]
		return token_dict, lengths
	
	#=============================================================
	@staticmethod
	def itersents(self,conllu_file):
		""""""
		
		if conllu_file.endswith('.zip'):
			open_func = zipfile.Zipfile
			kwargs = {}
		elif conllu_file.endswith('.gz'):
			open_func = gzip.open
			kwargs = {}
		elif conllu_file.endswith('.xz'):
			open_func = lzma.open
			kwargs = {'errors': 'ignore'}
		else:
			# open_func = codecs.open
			open_func = open
			kwargs = {'errors': 'ignore'}
		with open(conllu_file,encoding='utf8') as f:
			# with open_func(conllu_file, 'rb') as f:
			#reader = codecs.getreader('utf-8')(f, **kwargs)
			reader=f.readlines()
			buff = []
			for line in reader:
				line = line.strip()
				if line.startswith('#'):
					self.id_buff.append(line)
				if line and not line.startswith('#'):
					if not re.match('[0-9]+[-.][0-9]+', line):
						buff.append(line.split('\t'))
						#buff.append(line.split())
				elif buff:
					yield buff
					buff = []
			yield buff
	
	#=============================================================
	@property
	def n_sents(self):
		return len(self._lengths)
	@property
	def save_dir(self):
		return self._config.getstr(self, 'save_dir')
	@property
	def conllu_files(self):
		return list(self._conllu_files)
	@property
	def max_buckets(self):
		return self._config.getint(self, 'max_buckets')
	@property
	def batch_size(self):
		return self._config.getint(self, 'batch_size')
	@property
	def classname(self):
		return self.__class__.__name__
	
	#=============================================================
	def __enter__(self):
		return self
	def __exit__(self, exception_type, exception_value, trace):
		if exception_type is not None:
			raise
		self.close()
		return

#***************************************************************
class CoNLLUTrainset(CoNLLUDataset):
	def __init__(self, *args, config=None, **kwargs):
		self.setclass='train'
		super(CoNLLUTrainset, self).__init__(config.getfiles(self, 'train_conllus'), *args, config=config, **kwargs)
		
class CoNLLUDevset(CoNLLUDataset):
	def __init__(self, *args, config=None, **kwargs):
		self.setclass='dev'
		super(CoNLLUDevset, self).__init__(config.getfiles(self, 'dev_conllus'), *args, config=config, **kwargs)
		
class CoNLLUTestset(CoNLLUDataset):
	def __init__(self, *args, config=None, **kwargs):
		self.setclass='test'
		super(CoNLLUTestset, self).__init__(config.getfiles(self, 'test_conllus'), *args, config=config, **kwargs)
		
