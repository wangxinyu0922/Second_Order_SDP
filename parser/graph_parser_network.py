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

import re
import os
import pickle as pkl
import curses
import codecs

import numpy as np
import tensorflow as tf

from parser.base_network import BaseNetwork
from parser.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser.structs.vocabs.pointer_generator import PointerGenerator
import pdb
#***************************************************************
class GraphParserNetwork(BaseNetwork):
	""""""
	
	#=============================================================
	def build_graph(self, input_network_outputs={}, reuse=True, debug=False, nornn=False):
		""""""
		#pdb.set_trace()
		with tf.variable_scope('Embeddings'):

			if self.sum_pos: # TODO this should be done with a `POSMultivocab`
				pos_vocabs = list(filter(lambda x: 'POS' in x.classname, self.input_vocabs))
				pos_tensors = [input_vocab.get_input_tensor(embed_keep_prob=1, reuse=reuse) for input_vocab in pos_vocabs]
				non_pos_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs if 'POS' not in input_vocab.classname]
				#pos_tensors = [tf.Print(pos_tensor, [pos_tensor]) for pos_tensor in pos_tensors]
				#non_pos_tensors = [tf.Print(non_pos_tensor, [non_pos_tensor]) for non_pos_tensor in non_pos_tensors]
				if pos_tensors:
					pos_tensors = tf.add_n(pos_tensors)
					if not reuse:
						pos_tensors = [pos_vocabs[0].drop_func(pos_tensors, pos_vocabs[0].embed_keep_prob)]
					else:
						pos_tensors = [pos_tensors]
				input_tensors = non_pos_tensors + pos_tensors
			else:#run this
				input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]
			for input_network, output in input_network_outputs:
				with tf.variable_scope(input_network.classname):
					input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
			layer = tf.concat(input_tensors, 2)#batch*sentence*feature? or batch* sentence^2*feature?
		#pdb.set_trace()
		n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keepdims=True))
		batch_size, bucket_size, input_size = nn.get_sizes(layer)
		layer *= input_size / (n_nonzero + tf.constant(1e-12))
		

		token_weights = nn.greater(self.id_vocab.placeholder, 0)#find sentence length
		tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
		seq_lengths = tokens_per_sequence+1#batch size list of sentence length
		if self.use_seq2seq:
			token_weights = nn.greater(self.node_id_vocab.placeholder, 0)#find sentence length
			bucket_size = tf.shape(self.node_id_vocab.placeholder)[1]
			tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
			node_lengths = tokens_per_sequence+2# for rnn decoder


			# here we remove the the <bos> token for simplicity
			token_weights = nn.greater(self.node_id_vocab.placeholder[:,1:-1], 0)#find sentence length
			bucket_size = tf.shape(token_weights)[1]
			tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
		n_tokens = tf.reduce_sum(tokens_per_sequence)
		n_sequences = tf.count_nonzero(tokens_per_sequence)
		
		#pdb.set_trace()
		root_weights = token_weights + (1-nn.greater(tf.range(bucket_size), 0))
		token_weights3D = tf.expand_dims(token_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
		token_weights2D = tf.expand_dims(root_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
		# as our three dimension a b c, is a->b to deciding, so all binary potential should not contain root(x)
		# in fact root should contained in second order prediction except sibling, but for simpler we set all for same 
		token_weights4D = tf.cast(tf.expand_dims(token_weights2D, axis=-3) * tf.expand_dims(tf.expand_dims(root_weights, axis=-1),axis=-1),dtype=tf.float32)
		# abc -> ab,ac
		#token_weights_sib = tf.cast(tf.expand_dims(root_, axis=-3) * tf.expand_dims(tf.expand_dims(root_weights, axis=-1),axis=-1),dtype=tf.float32)
		#abc -> ab,cb
		#pdb.set_trace()
		token_weights_cop = tf.cast(tf.expand_dims(token_weights2D, axis=-2) * tf.expand_dims(tf.expand_dims(token_weights, axis=1),axis=-1),dtype=tf.float32)
		token_weights_cop_0 = token_weights_cop[:,0] * tf.cast(tf.transpose(token_weights3D,[0,2,1]),dtype=tf.float32)
		token_weights_cop = tf.concat([token_weights_cop_0[:,None,:],token_weights_cop[:,1:]],1)
		#data=np.stack((devprint['printdata']['layer_cop'][0][0]*devprint['token_weights3D'][0].T)[None,:],devprint['printdata']['layer_cop'][0][1:])
		#abc -> ab, bc
		token_weights_gp = tf.cast(tf.expand_dims(tf.transpose(token_weights3D,[0,2,1]), axis=-1) * tf.expand_dims(tf.expand_dims(token_weights, axis=1),axis=1),dtype=tf.float32)
		#abc -> ca, ab
		token_weights_gp2 = tf.cast(tf.expand_dims(token_weights3D, axis=2) * tf.expand_dims(tf.expand_dims(token_weights, axis=-1),axis=1),dtype=tf.float32)
		token_weights_sib = token_weights_gp
		#token_weights4D = tf.expand_dims(token_weights3D, axis=-3) * tf.expand_dims(tf.expand_dims(token_weights, axis=-1),axis=-1)
		tokens = {'n_tokens': n_tokens,
							'tokens_per_sequence': tokens_per_sequence,
							'token_weights': token_weights,
							'token_weights3D': token_weights,
							'n_sequences': n_sequences}
		
		conv_keep_prob = 1. if reuse else self.conv_keep_prob
		recur_keep_prob = 1. if reuse else self.recur_keep_prob
		recur_include_prob = 1. if reuse else self.recur_include_prob
		#R=BiLSTM(X)
		# pdb.set_trace()
		for i in six.moves.range(self.n_layers):
			conv_width = self.first_layer_conv_width if not i else self.conv_width
			#'''
			if not nornn and not self.nornn:
					with tf.variable_scope('RNN-{}'.format(i)):
						layer, sentence_feat = recurrent.directed_RNN(layer, self.recur_size, seq_lengths,
																							bidirectional=self.bidirectional,
																							recur_cell=self.recur_cell,
																							conv_width=conv_width,
																							recur_func=self.recur_func,
																							conv_keep_prob=conv_keep_prob,
																							recur_include_prob=recur_include_prob,
																							recur_keep_prob=recur_keep_prob,
																							cifg=self.cifg,
																							highway=self.highway,
																							highway_func=self.highway_func,
																							bilin=self.bilin)
			#'''
		if self.separate_prediction:
			print('separating the whole two pipeline')
			with tf.device('/device:GPU:1'):  
				for i in six.moves.range(self.n_layers):
					conv_width = self.first_layer_conv_width if not i else self.conv_width
					#'''
					if not nornn and not self.nornn:
							with tf.variable_scope('RNN2-{}'.format(i)):
								layer_rel, sentence_feat = recurrent.directed_RNN(layer, self.recur_size, seq_lengths,
																									bidirectional=self.bidirectional,
																									recur_cell=self.recur_cell,
																									conv_width=conv_width,
																									recur_func=self.recur_func,
																									conv_keep_prob=conv_keep_prob,
																									recur_include_prob=recur_include_prob,
																									recur_keep_prob=recur_keep_prob,
																									cifg=self.cifg,
																									highway=self.highway,
																									highway_func=self.highway_func,
																									bilin=self.bilin)
		else:
			layer_rel=layer
		#pdb.set_trace()
		output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
		outputs = {}
		#parser/structs/vocabs/token_vocabs.py loss is calculated in get_...
		# pdb.set_trace()
		# for seq2seq, create new features
		if 'correspond_word' in output_fields:
			print('use seq2seq model for node prediction')
			with tf.variable_scope('Seq2SeqDecoder'):
				sequence_length={}
				# here we remove 'root' node in the source sentence.(x)
				sequence_length['source']=seq_lengths
				sequence_length['target']=node_lengths
				pos_vocabs=[]
				lemma_vocabs=[]
				for input_vocab in self.input_vocabs:
					if 'POS' in input_vocab.classname:
						pos_vocabs.append(input_vocab.get_input_tensor(reuse=reuse))
					if 'Lemma' in input_vocab.classname:
						lemma_vocabs.append(input_vocab.get_input_tensor(reuse=reuse))
				reinput_tensors = pos_vocabs+lemma_vocabs
				reinput_tensors = tf.concat(reinput_tensors, 2)
				seq2seq_input_tensors = []
				if len(self.decoder_vocabs)>0:
					# get node label embedding
					seq2seq_input_tensors = [decoder_vocab.get_input_tensor(reuse=reuse) for decoder_vocab in self.decoder_vocabs if 'Copy' not in decoder_vocab.classname]
					# pdb.set_trace()
					pointer_generator_inputs = {decoder_vocab.classname[7:-5]:decoder_vocab.placeholder for decoder_vocab in self.decoder_vocabs if 'Copy' in decoder_vocab.classname}
					input_shape = seq2seq_input_tensors[0].shape
				target_placeholder=output_fields['correspond_word'].placeholder
				#target_tensor_shape=[input_shape[0],input_shape[1],reinput_tensors.shape[-1]]
				mapping = nn.greater(target_placeholder,0)
				result_tensor=tf.batch_gather(reinput_tensors,mapping)*tf.cast((mapping>0),dtype=tf.float32)[:,:,None]
				input_features = seq2seq_input_tensors + [result_tensor]
				input_feature = tf.concat(input_features, 2)

				#input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]
				#pdb.set_trace()
				# [batch, num_sequence_tokens+1, hidden], [batch, num_node_tokens+2,hidden] -> [batch, num_node_tokens+2, num_sequence_tokens+1]
				# here remove the "root" node from sentence encoder, so the output layer is one sequence smaller, and the mask "token_weights3D" should be smaller as well(x)
				node_encoding=output_fields['correspond_word'].forward(layer,input_feature[:,:-1],sentence_feat,token_weights3D,sequence_length)
				
				if 'label' in output_fields:
					# pdb.set_trace()
					self._evals.add('label')
					label_vocab=output_fields['label']
					label_vocab.predictor = PointerGenerator(label_vocab.hidden_size, label_vocab.hidden_size, len(label_vocab), 0, True, label_vocab.hidden_func, label_vocab.hidden_keep_prob)
					node_outputs=label_vocab.forward(node_encoding['values'], node_encoding['SrcWeights'], node_encoding['CorefWeights'], pointer_generator_inputs, debug=debug)
					outputs['label']=node_outputs
				# pdb.set_trace()
				# remove the start and end token
				
				layer=node_encoding['values'][:,:-1]
				layer_rel=layer
				# pdb.set_trace()
		#layers
		with tf.variable_scope('Classifiers'):
			if 'semrel' in output_fields:
				vocab = output_fields['semrel']
				head_vocab = output_fields['semhead']
				head_vocab.token_weights_sib=token_weights_sib
				head_vocab.token_weights_cop=token_weights_cop
				head_vocab.token_weights_gp=token_weights_gp
				head_vocab.token_weights_gp2=token_weights_gp2
				head_vocab.token_weights=token_weights
				if vocab.factorized:
					with tf.variable_scope('Unlabeled'):
						#pdb.set_trace()
						if self.layer_mask(head_vocab):
							unlabeled_outputs = head_vocab.get_bilinear_discriminator(
								layer,
								token_weights=token_weights3D,
								reuse=reuse, debug=debug, token_weights4D=token_weights4D)
						else:
							unlabeled_outputs = head_vocab.get_bilinear_discriminator(
								layer,
								token_weights=token_weights3D,
								reuse=reuse, debug=debug)

					if self.two_gpu:
						with tf.device('/device:GPU:1'):
							with tf.variable_scope('Labeled'):
								labeled_outputs = vocab.get_bilinear_classifier(
									layer_rel, unlabeled_outputs,
									token_weights=token_weights3D,
									reuse=reuse, debug=debug)
					else:
						with tf.variable_scope('Labeled'):
							labeled_outputs = vocab.get_bilinear_classifier(
								layer_rel, unlabeled_outputs,
								token_weights=token_weights3D,
								reuse=reuse, debug=debug)
				else:
					labeled_outputs = vocab.get_unfactored_bilinear_classifier(layer, head_vocab.placeholder,
						token_weights=token_weights3D,
						reuse=reuse)
				outputs['semgraph'] = labeled_outputs
				self._evals.add('semgraph')
			elif 'semhead' in output_fields:
				vocab = output_fields['semhead']
				outputs[vocab.classname] = vocab.get_bilinear_classifier(
					layer,
					token_weights=token_weights3D,
					reuse=reuse)
				self._evals.add('semhead')
			if 'attr' in output_fields:
					print('predict attributes')
					attr_vocab = output_fields['attr']
					with tf.variable_scope('Attribute'):
						attr_outputs = attr_vocab.get_bilinear_classifier(
							layer_rel, labeled_outputs,
							token_weights=token_weights[:,:,None],
							reuse=reuse, debug=debug)
					self._evals.add('attribute')
					outputs['attribute'] = attr_outputs
					# if 'semgraph' in outputs:
					# 	outputs['semgraph']['loss'] = tf.zeros(outputs['attribute']['loss'].shape,dtype=tf.float32)
			
			# -------------------------------------------------------------------------
			if 'frame' in output_fields:
				print('predict sdp frames')
				frame_vocab = output_fields['frame']
				with tf.variable_scope('Frame'):
					frame_outputs = frame_vocab.get_linear_classifier(layer_rel, token_weights, reuse=reuse, debug=debug)
				self._evals.add('frame')
				# pdb.set_trace()
				outputs['frame'] = frame_outputs
			# ---------------------------------------------------------------------------

		if debug:
			outputs['semgraph']['token_weights']=token_weights
			outputs['semgraph']['token_weights3D']=token_weights3D
			outputs['semgraph']['root_weights']=root_weights
			outputs['semgraph']['token_weights4D']=token_weights4D
			outputs['semgraph']['token_weights_sib']=token_weights_sib
			outputs['semgraph']['token_weights_cop']=token_weights_cop
			outputs['semgraph']['token_weights_gp']=token_weights_gp
			outputs['semgraph']['token_weights_gp2']=token_weights_gp2
			outputs['semgraph']['printdata']['word_postag']=self.input_vocabs[-1].placeholder
			if 'correspond_word' in output_fields:
				outputs['semgraph']['input_feature']=input_feature
				if debug:
					outputs['semgraph']['decoder']=node_encoding
					outputs['semgraph']['nodes']=node_outputs
			if 'ufeats' in output_fields:
				outputs['semgraph']['frame']=outputs['frame']
		return outputs, tokens
	
	#=============================================================
	def layer_mask(self, vocab):
		try:
			return self._config.getboolean(vocab, 'layer_mask')
		except:
			return False
	@property
	def sum_pos(self):
		return self._config.getboolean(self, 'sum_pos')
	@property
	def separate_prediction(self):
		try:
			return self._config.getboolean(self, 'separate_prediction')
		except:
			return False
	@property
	def two_gpu(self):
		try:
			return self._config.getboolean(self, 'two_gpu')
		except:
			return False
	@property
	def nornn(self):
		try:
			return self._config.getboolean(self, 'nornn')
		except:
			return False
	@property
	def predict_attribute(self):
		try:
			return self._config.getboolean(self, 'predict_attribute')
		except:
			return False
	