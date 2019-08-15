#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2019 Xinyu Wang


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import numpy as np
import tensorflow as tf

from .base_vocabs import BaseVocab 
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, classifiers
import pdb
#***************************************************************
class SecondOrderLBPVocab(BaseVocab):
	""""""
	
	#=============================================================
	def __init__(self, *args, **kwargs):
		""""""
		
		super(SecondOrderLBPVocab, self).__init__(*args, **kwargs)
		
		self.PAD_STR = '_'
		self.PAD_IDX = -1
		self.ROOT_STR = '0'
		self.ROOT_IDX = 0
		return
	
	#=============================================================
	def add(self, token):
		""""""
		
		return self.index(token)
	
	#=============================================================
	def token(self, index):
		""""""
		
		if index > -1:
			return str(index)
		else:
			return '_'
	
	#=============================================================
	def index(self, token):
		""""""
		
		if token != '_':
			return int(token)
		else:
			return -1
	
	#=============================================================
	def get_root(self):
		""""""
		
		return self.ROOT_STR
	
	#=============================================================
	def get_bilinear_classifier(self, layer, token_weights, variable_scope=None, reuse=False):
		""""""
		
		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		hidden_func = self.hidden_func
		hidden_size = self.hidden_size
		add_linear = self.add_linear
		linearize = self.linearize
		distance = self.distance
		n_splits = 2*(1+linearize+distance)
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers-1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, n_splits*hidden_size,
																		hidden_func=hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, n_splits*[hidden_size],
																		hidden_func=hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)
			if linearize:
				lin_layer1, lin_layer2 = layers.pop(0), layers.pop(0)
			if distance:
				dist_layer1, dist_layer2 = layers.pop(0), layers.pop(0)
			
			with tf.variable_scope('Attention'):
				if self.diagonal:
					logits, _ = classifiers.diagonal_bilinear_attention(
						layer1, layer2, 
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
					if linearize:
						with tf.variable_scope('Linearization'):
							lin_logits = classifiers.diagonal_bilinear_discriminator(
								lin_layer1, lin_layer2,
								hidden_keep_prob=hidden_keep_prob,
								add_linear=add_linear)
					if distance:
						with tf.variable_scope('Distance'):
							dist_lamda = 1+tf.nn.softplus(classifiers.diagonal_bilinear_discriminator(
								dist_layer1, dist_layer2,
								hidden_keep_prob=hidden_keep_prob,
								add_linear=add_linear))
				else:
					logits, _ = classifiers.bilinear_attention(
						layer1, layer2,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
					if linearize:
						with tf.variable_scope('Linearization'):
							lin_logits = classifiers.bilinear_discriminator(
								lin_layer1, lin_layer2,
								hidden_keep_prob=hidden_keep_prob,
								add_linear=add_linear)
					if distance:
						with tf.variable_scope('Distance'):
							dist_lamda = 1+tf.nn.softplus(classifiers.bilinear_discriminator(
								dist_layer1, dist_layer2,
								hidden_keep_prob=hidden_keep_prob,
								add_linear=add_linear))
				
				#-----------------------------------------------------------
				# Process the targets
				targets = self.placeholder
				shape = tf.shape(layer1)
				batch_size, bucket_size = shape[0], shape[1]
				# (1 x m)
				ids = tf.expand_dims(tf.range(bucket_size), 0)
				# (1 x m) -> (1 x 1 x m)
				head_ids = tf.expand_dims(ids, -2)
				# (1 x m) -> (1 x m x 1)
				dep_ids = tf.expand_dims(ids, -1)
				if linearize:
					# Wherever the head is to the left
					# (n x m), (1 x m) -> (n x m)
					lin_targets = tf.to_float(tf.less(targets, ids))
					# cross-entropy of the linearization of each i,j pair
					# (1 x 1 x m), (1 x m x 1) -> (n x m x m)
					lin_ids = tf.tile(tf.less(head_ids, dep_ids), [batch_size, 1, 1])
					# (n x 1 x m), (n x m x 1) -> (n x m x m)
					lin_xent = -tf.nn.softplus(tf.where(lin_ids, -lin_logits, lin_logits))
					# add the cross-entropy to the logits
					# (n x m x m), (n x m x m) -> (n x m x m)
					logits += tf.stop_gradient(lin_xent)
				if distance:
					# (n x m) - (1 x m) -> (n x m)
					dist_targets = tf.abs(targets - ids)
					# KL-divergence of the distance of each i,j pair
					# (1 x 1 x m) - (1 x m x 1) -> (n x m x m)
					dist_ids = tf.to_float(tf.tile(tf.abs(head_ids - dep_ids), [batch_size, 1, 1]))+1e-12
					# (n x m x m), (n x m x m) -> (n x m x m)
					#dist_kld = (dist_ids * tf.log(dist_lamda / dist_ids) + dist_ids - dist_lamda)
					dist_kld = -tf.log((dist_ids - dist_lamda)**2/2 + 1)
					# add the KL-divergence to the logits
					# (n x m x m), (n x m x m) -> (n x m x m)
					logits += tf.stop_gradient(dist_kld)
				
				#-----------------------------------------------------------
				# Compute probabilities/cross entropy
				# (n x m) + (m) -> (n x m)
				non_pads = tf.to_float(token_weights) + tf.to_float(tf.logical_not(tf.cast(tf.range(bucket_size), dtype=tf.bool)))
				# (n x m x m) o (n x 1 x m) -> (n x m x m)
				probabilities = tf.nn.softmax(logits) * tf.expand_dims(non_pads, -2)
				# (n x m), (n x m x m), (n x m) -> ()
				loss = tf.losses.sparse_softmax_cross_entropy(
					targets,
					logits,
					weights=token_weights)
				# (n x m) -> (n x m x m x 1)
				one_hot_targets = tf.expand_dims(tf.one_hot(targets, bucket_size), -1)
				# (n x m) -> ()
				n_tokens = tf.to_float(tf.reduce_sum(token_weights))
				if linearize:
					# (n x m x m) -> (n x m x 1 x m)
					lin_xent_reshaped = tf.expand_dims(lin_xent, -2)
					# (n x m x 1 x m) * (n x m x m x 1) -> (n x m x 1 x 1)
					lin_target_xent = tf.matmul(lin_xent_reshaped, one_hot_targets)
					# (n x m x 1 x 1) -> (n x m)
					lin_target_xent = tf.squeeze(lin_target_xent, [-1, -2])
					# (n x m), (n x m), (n x m) -> ()
					loss -= tf.reduce_sum(lin_target_xent*tf.to_float(token_weights)) / (n_tokens + 1e-12)
				if distance:
					# (n x m x m) -> (n x m x 1 x m)
					dist_kld_reshaped = tf.expand_dims(dist_kld, -2)
					# (n x m x 1 x m) * (n x m x m x 1) -> (n x m x 1 x 1)
					dist_target_kld = tf.matmul(dist_kld_reshaped, one_hot_targets)
					# (n x m x 1 x 1) -> (n x m)
					dist_target_kld = tf.squeeze(dist_target_kld, [-1, -2])
					# (n x m), (n x m), (n x m) -> ()
					loss -= tf.reduce_sum(dist_target_kld*tf.to_float(token_weights)) / (n_tokens + 1e-12)
				
				#-----------------------------------------------------------
				# Compute predictions/accuracy
				# (n x m x m) -> (n x m)
				predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
				# (n x m) (*) (n x m) -> (n x m)
				correct_tokens = nn.equal(targets, predictions) * token_weights
				# (n x m) -> (n)
				tokens_per_sequence = tf.reduce_sum(token_weights, axis=-1)
				# (n x m) -> (n)
				correct_tokens_per_sequence = tf.reduce_sum(correct_tokens, axis=-1)
				# (n), (n) -> (n)
				correct_sequences = nn.equal(tokens_per_sequence, correct_tokens_per_sequence)
		
		#-----------------------------------------------------------
		# Populate the output dictionary
		outputs = {}
		outputs['recur_layer'] = recur_layer
		outputs['unlabeled_targets'] = self.placeholder
		outputs['probabilities'] = probabilities
		outputs['unlabeled_loss'] = loss
		outputs['loss'] = loss
		
		outputs['unlabeled_predictions'] = predictions
		outputs['predictions'] = predictions
		outputs['correct_unlabeled_tokens'] = correct_tokens
		outputs['n_correct_unlabeled_tokens'] = tf.reduce_sum(correct_tokens)
		outputs['n_correct_unlabeled_sequences'] = tf.reduce_sum(correct_sequences)
		outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
		outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
		return outputs
	
	#=============================================================
	def __getitem__(self, key):
		if isinstance(key, six.string_types):
			if key == '_':
				return -1
			else:
				return int(key)
		elif isinstance(key, six.integer_types + (np.int32, np.int64)):
			if key > -1:
				return str(key)
			else:
				return '_'
		elif hasattr(key, '__iter__'):
			return [self[k] for k in key]
		else:
			raise ValueError('key to IndexVocab.__getitem__ must be (iterable of) string or integer')
		return
	#=============================================================
	@property
	def distance(self):
		return self._config.getboolean(self, 'distance')
	@property
	def linearize(self):
		return self._config.getboolean(self, 'linearize')
	@property
	def decomposition_level(self):
		return self._config.getint(self, 'decomposition_level')
	@property
	def diagonal(self):
		return self._config.getboolean(self, 'diagonal')
	@property
	def add_linear(self):
		return self._config.getboolean(self, 'add_linear')
	@property
	def n_layers(self):
		return self._config.getint(self, 'n_layers')
	@property
	def hidden_size(self):
		return self._config.getint(self, 'hidden_size')
	@property
	def hidden_keep_prob(self):
		return self._config.getfloat(self, 'hidden_keep_prob')
	@property
	def hidden_keep_prob_tri(self):
		try:
			return self._config.getfloat(self, 'hidden_keep_prob_tri')
		except:
			return self._config.getfloat(self, 'hidden_keep_prob')
		
	@property
	def hidden_func(self):
		hidden_func = self._config.getstr(self, 'hidden_func')
		if hasattr(nonlin, hidden_func):
			return getattr(nonlin, hidden_func)
		else:
			raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
	@property
	def num_iteration(self):
		return self._config.getfloat(self, 'num_iteration')
	@property
	def discriminator2(self):
		return self._config.getboolean(self,'discriminator2')
	@property
	def sibling_only(self):
		return self._config.getboolean(self,'sibling_only')
	@property
	def self_minus(self):
		return self._config.getboolean(self,'self_minus')
	@property
	def use_sib(self):
		return self._config.getboolean(self,'use_sib')
	@property
	def use_gp(self):
		return self._config.getboolean(self,'use_gp')
	@property
	def use_cop(self):
		return self._config.getboolean(self,'use_cop')
	@property
	def transposed(self):
		return self._config.getboolean(self,'transposed')
	@property
	def unary_weight(self):
		if self._config.getboolean(self,'unary_weight'):
			return int(self.use_cop)+int(self.use_sib)+2*int(self.use_gp)
		else:
			return 1
	@property
	def new_potential(self):
		return self._config.getboolean(self,'new_potential')
	@property
	def separate_embed(self):
		try:
			return self._config.getboolean(self,'separate_embed')
		except:
			return False
	@property
	def old_trilin(self):
		try:
			return self._config.getboolean(self,'old_trilin')
		except:
			return False
	@property
	def remove_loop(self):
		try:
			return self._config.getboolean(self,'remove_loop')
		except:
			return False
	@property
	def combine_loss(self):
		try:
			return self._config.getboolean(self,'combine_loss')
		except:
			return False
	@property
	def loss_weight(self):
		try:
			return self._config.getfloat(self,'loss_weight')
		except:
			return 0.5
	@property
	def loss_weight_unary(self):
		try:
			return self._config.getfloat(self,'loss_weight_unary')
		except:
			return 0.5
	@property
	def test_new_potential(self):
		try:
			return self._config.getboolean(self,'test_new_potential')
		except:
			return False
	@property
	def layer_mask(self):
		try:
			return self._config.getboolean(self,'layer_mask')
		except:
			return False
	@property
	def normalize(self):
		try:
			return self._config.getboolean(self,'normalize')
		except:
			return False
	@property
	def use_unary_hidden(self):
		try:
			return self._config.getboolean(self,'use_unary_hidden')
		except:
			return False
	@property
	def unary_hidden(self):
		try:
			return self._config.getint(self,'unary_hidden')
		except:
			return self._config.getint(self,'hidden_size')
	@property
	def tri_std(self):
		try:
			return self._config.getfloat(self,'tri_std')
		except:
			return 0.01
	@property
	def tri_std_unary(self):
		try:
			return self._config.getfloat(self,'tri_std_unary')
		except:
			return 0.5
	@property
	def hidden_k(self):
		try:
			return self._config.getint(self,'hidden_k')
		except:
			return 200
	@property
	def two_gpu(self):
		try:
			if self._config.get('GraphParserNetwork','two_gpu')=='True':
				return True
			else:
				return False
		except:
			return False
	@property
	def remove_root_child(self):
		try:
			return self._config.getboolean(self,'remove_root_child')
		except:
			return False
	@property
	def compare_precision(self):
		try:
			if self._config.get('DEFAULT', 'tb')=='ptb' or self._config.get('DEFAULT', 'tb')=='ctb':
				return True
			else:
				return False
		except:
			return False
#***************************************************************
class GraphSecondLBPVocab(SecondOrderLBPVocab): #second order trilinear classifier
	""""""
	
	_depth = -1
	
	#=============================================================
	def __init__(self, *args, **kwargs):
		""""""
		
		kwargs['placeholder_shape'] = [None, None, None]
		super(GraphSecondLBPVocab, self).__init__(*args, **kwargs)
		return
	
	#=============================================================
	def get_bilinear_discriminator(self, layer, token_weights, variable_scope=None, reuse=False, debug=False, token_weights4D=None):
		""""""
		#in fact here is get_trilinear_discriminator
		outputs = {}
		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		hidden_keep_prob_tri = 1 if reuse else self.hidden_keep_prob_tri
		add_linear = self.add_linear
		#here set n_splits to be three
		if self.separate_embed:
			n_splits = 9*(1+self.linearize+self.distance)
		else:
			n_splits = 3*(1+self.linearize+self.distance)
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers-1):#number of layers of FNN? what is this?
				with tf.variable_scope('FC-%d' % i):#here is FNN? did not run
					layer = classifiers.hidden(layer, n_splits*self.hidden_size,
												 hidden_func=self.hidden_func,
												 hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):#FNN output and split two layer? FNN+split. Linear transform for a sentence, n_splits is number of features you want 
				#this linear transformation contains word information
				if self.use_unary_hidden:
					print('separate unary and binary hidden size')
					if self.separate_embed:
						hidden_list=2*[self.unary_hidden]+(n_splits-2)*[self.hidden_size]
					else:
						hidden_list=2*[self.unary_hidden]+n_splits*[self.hidden_size]
				else:
					hidden_list=n_splits*[self.hidden_size]
				#pdb.set_trace()
				layers = classifiers.hiddens(layer, hidden_list,
											 hidden_func=self.hidden_func,
											 hidden_keep_prob=hidden_keep_prob)
			if self.separate_embed:
				# unary_head + unary_dep + sib_head + sib_dep + gp_head + gp_dep + gp_(head+dep) + cop_head + cop_dep
				unary_layer1, unary_layer2, sib_head, sib_dep, gp_head, gp_dep, gp_headdep, cop_head, cop_dep = layers.pop(0), layers.pop(0), layers.pop(0), layers.pop(0), layers.pop(0)\
																																													, layers.pop(0), layers.pop(0), layers.pop(0), layers.pop(0)
			else:
				# head + dep + (head+dep)
				if self.use_unary_hidden:
					unary_layer1, unary_layer2, layer1, layer2, layer3 = layers.pop(0), layers.pop(0), layers.pop(0), layers.pop(0), layers.pop(0)
				else:
					layer1, layer2, layer3 = layers.pop(0), layers.pop(0), layers.pop(0)
					#pdb.set_trace()
					unary_layer1=layer1
					unary_layer2=layer2
			if self.linearize:#false
				lin_layer1, lin_layer2 = layers.pop(0), layers.pop(0), layers.pop(0)
			if self.distance:#false in graph
				dist_layer1, dist_layer2 = layers.pop(0), layers.pop(0), layers.pop(0)
			#pdb.set_trace()
			if self.layer_mask:
				#pdb.set_trace()
				sentence_mask=tf.expand_dims(tf.cast(tf.transpose(token_weights,[0,2,1])[:,0],dtype=tf.float32),-1)
				
				unary_layer1=unary_layer1*sentence_mask
				unary_layer2=unary_layer2*sentence_mask
				if self.separate_embed:
					sib_head, sib_dep, gp_head, gp_dep, gp_headdep, cop_head, cop_dep = sib_head*sentence_mask, sib_dep*sentence_mask,\
																						gp_head*sentence_mask, gp_dep*sentence_mask,\
																						gp_headdep*sentence_mask, cop_head*sentence_mask,\
																						cop_dep*sentence_mask
				else:
					if not self.separate_embed:
						layer1=layer1*sentence_mask
						layer2=layer2*sentence_mask
					layer3=layer3*sentence_mask
				pass
			with tf.variable_scope('Discriminator'):
				if self.diagonal:
					logits = classifiers.diagonal_bilinear_discriminator(
						layer1, layer2,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
					if self.linearize:
						with tf.variable_scope('Linearization'):
							lin_logits = classifiers.diagonal_bilinear_discriminator(
								lin_layer1, lin_layer2,
								hidden_keep_prob=hidden_keep_prob,
								add_linear=add_linear)
					if self.distance:
						with tf.variable_scope('Distance'):
							dist_lamda = 1+tf.nn.softplus(classifiers.diagonal_bilinear_discriminator(
								dist_layer1, dist_layer2,
								hidden_keep_prob=hidden_keep_prob,
								add_linear=add_linear))
				else:
					#only run here
					#First order potential and second order potential
					#change it to two label discriminator, because we need a score for the condition of 0 and 1
					#(n x m x 2 x m)

					#with tf.device('/device:GPU:1'):
					'''
					unary, unary_weights = classifiers.bilinear_classifier(
						layer1, layer2, 2,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
					'''
					unary = classifiers.bilinear_classifier(
						unary_layer1, unary_layer2, 2,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear,target_model='LBP', tri_std=self.tri_std_unary)
					#'''
					'''
					unary = classifiers.bilinear_discriminator(
						layer1, layer2,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
					#'''

						
					if self.separate_embed:
						print('separate')
						# head dep dep
						if self.test_new_potential:
							print('testing new potential')
							if self.use_sib:
								with tf.variable_scope('Sibling'):
									layer_sib = classifiers.trilinear_discriminator_outer(
										sib_head, sib_dep, sib_dep,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
							# head head+dep dep
							if self.use_gp:
								with tf.variable_scope('GrandParents'):
									layer_gp = classifiers.trilinear_discriminator_outer(
										gp_head, gp_headdep, gp_dep,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
							# head dep head
							if self.use_cop:
								with tf.variable_scope('CoParents'):
									layer_cop = classifiers.trilinear_discriminator_outer(
										cop_head, cop_dep, cop_head,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
						else:
							if self.use_sib:
								with tf.variable_scope('Sibling'):
									layer_sib = classifiers.trilinear_discriminator_new(
										sib_head, sib_dep, sib_dep,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
							if self.use_gp:
								with tf.variable_scope('GrandParents'):
									layer_gp= classifiers.trilinear_discriminator_new(
										gp_head, gp_headdep, gp_dep,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
									#'''
							# head dep head
							if self.use_cop:
								with tf.variable_scope('CoParents'):
									layer_cop = classifiers.trilinear_discriminator_new(
										cop_head, cop_dep, cop_head,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
									#'''
					elif not self.old_trilin:
						print('head dep')
						if self.test_new_potential:
							print('testing new potential')
							# head dep dep
							if self.use_sib:
								with tf.variable_scope('Sibling'):
									layer_sib = classifiers.trilinear_discriminator_test(
										layer1, layer2, layer2,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
							# head head+dep dep
							if self.use_gp:
								with tf.variable_scope('GrandParents'):
									layer_gp = classifiers.trilinear_discriminator_test(
										layer1, layer3, layer2,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
							# head dep head
							if self.use_cop:
								with tf.variable_scope('CoParents'):
									layer_cop = classifiers.trilinear_discriminator_test(
										layer1, layer2, layer1,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
						#=================================================
						else:
							# head dep dep
							if self.use_sib:
								with tf.variable_scope('Sibling'):
									layer_sib = classifiers.trilinear_discriminator_new(
										layer1, layer2, layer2,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
							# head head+dep dep
							if self.use_gp:
								with tf.variable_scope('GrandParents'):
									layer_gp = classifiers.trilinear_discriminator_new(
										layer1, layer3, layer2,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
							# head dep head
							if self.use_cop:
								with tf.variable_scope('CoParents'):
									layer_cop = classifiers.trilinear_discriminator_new(
										layer1, layer2, layer1,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear,target_model='LBP',tri_std=self.tri_std)
					#-----------------------------------------------------------
					if self.linearize:
						with tf.variable_scope('Linearization'):
							lin_logits = classifiers.bilinear_discriminator(
								lin_layer1, lin_layer2,
								hidden_keep_prob=hidden_keep_prob,
								add_linear=add_linear)
					if self.distance:
						with tf.variable_scope('Distance'):
							dist_lamda = 1+tf.nn.softplus(classifiers.bilinear_discriminator(
								dist_layer1, dist_layer2,
								hidden_keep_prob=hidden_keep_prob,
								add_linear=add_linear))
				
				binary_shape = layer_sib.shape.as_list()
				if debug:
					outputs['printdata']={}
					if self.new_potential:
						outputs['printdata']['unary_old']=unary
					if self.use_sib:
						outputs['printdata']['layer_sib_old']=layer_sib
					if self.use_cop:
						outputs['printdata']['layer_cop_old']=layer_cop
					if self.use_gp:
						outputs['printdata']['layer_gp_old']=layer_gp

				#======================LBP==========================
				
				#pdb.set_trace()
				#update: use log space for LBP
				if self.two_gpu:
					print('two gpu training for LBP')
					#pdb.set_trace()
					GPUID=1
				else:
					GPUID=0
				with tf.device('/device:GPU:'+str(GPUID)):  
					if self.use_sib:
						layer_type=layer_sib
					elif self.use_cop:
						layer_type=layer_cop
					elif self.use_gp:
						layer_type=layer_gp
					# (n x 2 x ma x mb x mc) ab <- ac
					log_message_sib=tf.stack([tf.zeros_like(layer_type),tf.zeros_like(layer_type)],1)
					# (n x 2 x ma x mb x mc) ab <- cb
					log_message_cop=tf.stack([tf.zeros_like(layer_type),tf.zeros_like(layer_type)],1)
					# (n x 2 x ma x mb x mc) ab <- bc 
					log_message_gp1=tf.stack([tf.zeros_like(layer_type),tf.zeros_like(layer_type)],1)
					# (n x 2 x ma x mb x mc) ab <- ca 
					log_message_gp2=tf.stack([tf.zeros_like(layer_type),tf.zeros_like(layer_type)],1)
					#pdb.set_trace()
					#'''
					if self.layer_mask:
						print('use layer mask')
						unary=unary*tf.cast(tf.expand_dims(tf.transpose(token_weights,[0,2,1]),-2),dtype=tf.float32)
						if self.remove_root_child:
							# abc -> ab,ac
							if self.use_sib:
								layer_sib=layer_sib*self.token_weights_sib
								log_message_sib=log_message_sib*self.token_weights_sib[:,None,:,:,:]
							if self.use_cop:
								layer_cop=layer_cop*self.token_weights_cop
								log_message_cop=log_message_cop*self.token_weights_cop[:,None,:,:,:]
							if self.use_gp:
								layer_gp=layer_gp*self.token_weights_gp
								log_message_gp1=log_message_gp1*self.token_weights_gp[:,None,:,:,:]
								log_message_gp2=log_message_gp2*self.token_weights_gp2[:,None,:,:,:]
						else:
							if self.use_sib:
								layer_sib=layer_sib*token_weights4D
							if self.use_cop:
								layer_cop=layer_cop*token_weights4D
							if self.use_gp:
								layer_gp=layer_gp*token_weights4D
							token_weights4D=tf.expand_dims(token_weights4D,1)
							if self.use_sib:
								log_message_sib=log_message_sib*token_weights4D
							if self.use_cop:
								log_message_cop=log_message_cop*token_weights4D
							if self.use_gp:
								log_message_gp1=log_message_gp1*token_weights4D
								log_message_gp2=log_message_gp2*token_weights4D
					#'''
					#(n x m x 2 x m) -> (n x 2 x m x m)
					batch_size=nn.get_sizes(layer_sib)[0]
					unary=tf.transpose(unary,[0,2,1,3])
					if self.use_sib:
						layer_sib = layer_sib-tf.linalg.band_part(layer_sib,-1,0) + tf.transpose(tf.linalg.band_part(layer_sib,0,-1),perm=[0,1,3,2])
					if self.use_gp:
						layer_gp2 = tf.transpose(layer_gp,perm=[0,2,3,1])
					if self.use_cop:
						#(n x ma x mb x mc) -> (n x mb x ma x mc) 
						#in order to create a symmtric tensor on ma and mc
						layer_cop = tf.transpose(layer_cop,perm=[0,2,1,3])
						# first set lower triangle part to be zero, then assign the upper triangle part transposed to lower triangle part
						layer_cop = layer_cop - tf.linalg.band_part(layer_cop,-1,0) + tf.transpose(tf.linalg.band_part(layer_cop,0,-1),perm=[0,1,3,2])
						# Finally (n x mb x ma x mc) -> (n x ma x mb x mc)
						layer_cop = tf.transpose(layer_cop,perm=[0,2,1,3])
					for i in range(1,int(self.num_iteration)+1):
						#(n x ma x mb x mc x 2)
						prev_sib=log_message_sib
						prev_cop=log_message_cop
						prev_gp1=log_message_gp1
						prev_gp2=log_message_gp2
						#first gp is a->b->c second is c->a->b
						#(n x 2 x ma x mb x mc) -> (n x 2 x ma x mb)
						#TODO: add mask when message passing
						FP=tf.reduce_sum(prev_gp1,-1)+tf.reduce_sum(prev_gp2,-1)+tf.reduce_sum(prev_cop,-1)+tf.reduce_sum(prev_sib,-1)
						#FP=tf.reduce_sum(prev_gp1,-1)+tf.reduce_sum(prev_gp2,-1)
						if debug:
							outputs['printdata']['FP0'+str(i)]=FP
							outputs['printdata']['prev_sib'+str(i)]=prev_sib
							outputs['printdata']['prev_cop'+str(i)]=prev_cop
							outputs['printdata']['prev_gp1'+str(i)]=prev_gp1
							outputs['printdata']['prev_gp2'+str(i)]=prev_gp2
						# remove self loop and self repeat (a b a) (a b b)
						#'''
						FP=FP-tf.transpose(tf.linalg.diag_part(tf.transpose(prev_sib,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(prev_sib)
						FP=FP-tf.transpose(tf.linalg.diag_part(tf.transpose(prev_cop,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(prev_cop)
						FP=FP-tf.transpose(tf.linalg.diag_part(tf.transpose(prev_gp1,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(prev_gp1)
						FP=FP-tf.transpose(tf.linalg.diag_part(tf.transpose(prev_gp2,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(prev_gp2)
						#'''
						#TODO: softmax form
						FP_potential=tf.expand_dims(unary+FP,-1)
						if self.use_sib:
							#update sibling
							#pdb.set_trace()
							#(n x 2 x ma x mc) -> (n x 2 x ma x mc x 1) - (n x 2 x ma x mc x mb)-> (n x 2 x ma x mc x mb)-> (n x 2 x ma x mb x mc)
							sib_FP=tf.transpose(FP_potential-prev_sib,perm=[0,1,2,4,3])
							log_message_sib_0=tf.reduce_logsumexp(sib_FP,axis=1)
							#(n x ma x mb x mc) + (n x ma x mb x mc) -> (n x ma x mb x mc)
							log_message_sib_1=tf.reduce_logsumexp(tf.stack([sib_FP[:,0],sib_FP[:,1]+layer_sib],1),axis=1)
							#(n x ma x mb x mc) -> (n x 2 x ma x mb x mc)
							log_message_sib=tf.stack([log_message_sib_0,log_message_sib_1],axis=1)
						if self.use_cop:
							#update coparents
							#(n x 2 x mc x mb) -> (n x 2 x mc x mb x 1) - (n x 2 x mc x mb x ma)-> (n x 2 x mc x mb x ma)-> (n x 2 x ma x mb x mc)
							cop_FP=tf.transpose(FP_potential-prev_cop,perm=[0,1,4,3,2])
							log_message_cop_0=tf.reduce_logsumexp(cop_FP,axis=1)
							#(n x ma x mb x mc) + (n x ma x mb x mc) -> (n x ma x mb x mc)
							log_message_cop_1=tf.reduce_logsumexp(tf.stack([cop_FP[:,0],cop_FP[:,1]+layer_cop],1),axis=1)
							#(n x ma x mb x mc) -> (n x 2 x ma x mb x mc)
							log_message_cop=tf.stack([log_message_cop_0,log_message_cop_1],axis=1)
						if self.use_gp:
							#update gp1 (a->b->c)
							#(n x 2 x ma x mc) -> (n x 2 x mb x mc x 1) - (n x 2 x mb x mc x ma)-> (n x 2 x mb x mc x ma)-> (n x 2 x ma x mb x mc)
							# Note that here gp1 should minus prev_gp2 as we need to keep ab->bc is still representing a->b->c
							gp_FP=tf.transpose(FP_potential-prev_gp2,perm=[0,1,4,2,3])
							log_message_gp_0=tf.reduce_logsumexp(gp_FP,axis=1)
							#(n x ma x mb x mc) + (n x ma x mb x mc) -> (n x ma x mb x mc)
							log_message_gp_1=tf.reduce_logsumexp(tf.stack([gp_FP[:,0],gp_FP[:,1]+layer_gp],1),axis=1)
							#(n x ma x mb x mc) -> (n x 2 x ma x mb x mc)
							log_message_gp1=tf.stack([log_message_gp_0,log_message_gp_1],axis=1)

							#update gp2 (c->a->b)
							#(n x 2 x ma x mc) -> (n x 2 x mc x ma x 1) - (n x 2 x mc x ma x mb)-> (n x 2 x mc x ma x mb)-> (n x 2 x ma x mb x mc)
							gp2_FP=tf.transpose(FP_potential-prev_gp1,perm=[0,1,3,4,2])
							log_message_gp2_0=tf.reduce_logsumexp(gp2_FP,axis=1)
							#(n x ma x mb x mc) + (n x ma x mb x mc) -> (n x ma x mb x mc)
							log_message_gp2_1=tf.reduce_logsumexp(tf.stack([gp2_FP[:,0],gp2_FP[:,1]+layer_gp2],1),axis=1)
							#(n x ma x mb x mc) -> (n x 2 x ma x mb x mc)
							log_message_gp2=tf.stack([log_message_gp2_0,log_message_gp2_1],axis=1)
						if self.normalize:
							if i==1:
								print('normalize each LBP iteration')
							if self.use_sib:
								log_message_sib=tf.nn.log_softmax(log_message_sib,1)
							if self.use_cop:
								log_message_cop=tf.nn.log_softmax(log_message_cop,1)
							if self.use_gp:
								log_message_gp1=tf.nn.log_softmax(log_message_gp1,1)
								log_message_gp2=tf.nn.log_softmax(log_message_gp2,1)
						if self.layer_mask:
							if self.remove_root_child:
								# abc -> ab,ac
								if self.use_sib:
									layer_sib=layer_sib*self.token_weights_sib
								if self.use_cop:
									layer_cop=layer_cop*self.token_weights_cop
								if self.use_gp:
									layer_gp=layer_gp*self.token_weights_gp
								log_message_sib=log_message_sib*self.token_weights_sib[:,None,:,:,:]
								log_message_cop=log_message_cop*self.token_weights_cop[:,None,:,:,:]
								log_message_gp1=log_message_gp1*self.token_weights_gp[:,None,:,:,:]
								log_message_gp2=log_message_gp2*self.token_weights_gp2[:,None,:,:,:]
							else:
								log_message_sib=log_message_sib*token_weights4D
								log_message_cop=log_message_cop*token_weights4D
								log_message_gp1=log_message_gp1*token_weights4D
								log_message_gp2=log_message_gp2*token_weights4D
						if debug:
							'''
							outputs['printdata']['gp2_FP'+str(i)]=gp2_FP
							outputs['printdata']['FP_potential'+str(i)]=FP_potential
							outputs['printdata']['FP4'+str(i)]=FP
							outputs['printdata']['sib_FP'+str(i)]=sib_FP
							outputs['printdata']['cop_FP'+str(i)]=cop_FP
							outputs['printdata']['gp1_FP'+str(i)]=gp_FP
							outputs['printdata']['gp2_FP'+str(i)]=gp2_FP
							outputs['printdata']['message_sib_0'+str(i)]=log_message_sib_0
							outputs['printdata']['message_sib_1'+str(i)]=log_message_sib_1
							outputs['printdata']['message_sib'+str(i)]=log_message_sib
							outputs['printdata']['message_cop_0'+str(i)]=log_message_cop_0
							outputs['printdata']['message_cop_1'+str(i)]=log_message_cop_1
							outputs['printdata']['message_cop'+str(i)]=log_message_cop
							outputs['printdata']['message_gp1'+str(i)]=log_message_gp1
							outputs['printdata']['message_gp2'+str(i)]=log_message_gp2
							outputs['printdata']['message_gp1_0'+str(i)]=log_message_gp_0
							outputs['printdata']['message_gp1_1'+str(i)]=log_message_gp_1
							outputs['printdata']['message_gp2_0'+str(i)]=log_message_gp2_0
							outputs['printdata']['message_gp2_1'+str(i)]=log_message_gp2_1
							#'''
							pass

					log_belief=tf.reduce_sum(log_message_gp1,-1)+tf.reduce_sum(log_message_gp2,-1)+tf.reduce_sum(log_message_cop,-1)+tf.reduce_sum(log_message_sib,-1)
					#log_belief=tf.reduce_sum(log_message_gp1,-1)+tf.reduce_sum(log_message_gp2,-1)
					# remove self loop and self repeat (a b a) (a b b)
					#'''
					log_belief=log_belief-tf.transpose(tf.linalg.diag_part(tf.transpose(log_message_sib,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(log_message_sib)
					log_belief=log_belief-tf.transpose(tf.linalg.diag_part(tf.transpose(log_message_cop,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(log_message_cop)
					log_belief=log_belief-tf.transpose(tf.linalg.diag_part(tf.transpose(log_message_gp1,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(log_message_gp1)
					log_belief=log_belief-tf.transpose(tf.linalg.diag_part(tf.transpose(log_message_gp2,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(log_message_gp2)
					#'''
					log_belief=log_belief+unary
					#q_value=belief/tf.reduce_sum(belief,axis=1,keepdims=True)
					#calculate softmax loss later
					q_value=log_belief
					if debug:
						outputs['printdata']['message_sib_fin']=tf.reduce_sum(log_message_sib,-1)-tf.transpose(tf.linalg.diag_part(tf.transpose(log_message_sib,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(log_message_sib)
						outputs['printdata']['message_cop_fin']=tf.reduce_sum(log_message_cop,-1)-tf.transpose(tf.linalg.diag_part(tf.transpose(log_message_cop,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(log_message_cop)
						outputs['printdata']['message_gp1_fin']=tf.reduce_sum(log_message_gp1,-1)-tf.transpose(tf.linalg.diag_part(tf.transpose(log_message_gp1,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(log_message_gp1)
						outputs['printdata']['message_gp2_fin']=tf.reduce_sum(log_message_gp2,-1)-tf.transpose(tf.linalg.diag_part(tf.transpose(log_message_gp2,perm=[0,1,3,2,4])),perm=[0,1,3,2])-tf.linalg.diag_part(log_message_gp2)
					#======================LBP==========================

				
					#-----------------------------------------------------------
					# Process the targetsb
					# (n x m x m) -> (n x m x m)
					#here in fact is a graph, which is m*m representing the connection between each edge
					unlabeled_targets = self.placeholder#ground truth graph, what is self.placeholder?
					#USELESS
					shape = tf.shape(unary_layer1)
					batch_size, bucket_size = shape[0], shape[1]
					# (1 x m)
					ids = tf.expand_dims(tf.range(bucket_size), 0)
					# (1 x m) -> (1 x 1 x m)
					head_ids = tf.expand_dims(ids, -2)
					# (1 x m) -> (1 x m x 1)
					dep_ids = tf.expand_dims(ids, -1)
					#pdb.set_trace()
					if debug:
						#outputs['printdata']['logits']=logits
						outputs['printdata']['q_value']=q_value
						outputs['printdata']['unary']=unary
						#outputs['printdata']['binary']=binary
						outputs['printdata']['belief']=log_belief
						if self.use_sib:
							outputs['printdata']['message_sib']=log_message_sib
							outputs['printdata']['message_sib_0']=log_message_sib_0
							outputs['printdata']['layer_sib']=layer_sib
						if self.use_gp:
							#outputs['printdata']['second_temp_gp']=second_temp_gp
							outputs['printdata']['message_gp1']=log_message_gp1
							outputs['printdata']['message_gp2']=log_message_gp2
							outputs['printdata']['layer_gp']=layer_gp
						if self.use_cop:
							outputs['printdata']['layer_cop']=layer_cop
							outputs['printdata']['message_cop']=log_message_cop
						#outputs['printdata']['layer1']=layer1
						#outputs['printdata']['layer2']=layer2
						if not self.separate_embed:
							outputs['printdata']['layer3']=layer3
						outputs['printdata']['targets']=unlabeled_targets
						outputs['printdata']['token_weights']=token_weights
						if self.sibling_only:
							outputs['printdata']['binary_weights']=binary_weights 
							outputs['printdata']['binary']=layer_sib
						if self.new_potential:
								#outputs['printdata']['layer_sib2']=layer_sib2
								outputs['printdata']['layer_gp2']=layer_gp2
								#outputs['printdata']['layer_cop2']=layer_cop2
								pass
						#outputs['printdata']['binary_weights']=binary_weights
						'''
						outputs['printdata']['unary_weights']=unary_weights
						outputs['printdata']['binary_weights']=binary_weights
						outputs['printdata']['binary_weights_cop']=binary_weights_cop
						outputs['printdata']['binary_weights_gp']=binary_weights_gp
						#'''

					#no running here
					if self.linearize:
						# Wherever the head is to the left
						# (n x m x m), (1 x m x 1) -> (n x m x m)
						lin_targets = tf.to_float(tf.less(unlabeled_targets, dep_ids))
						# cross-entropy of the linearization of each i,j pair
						# (1 x 1 x m), (1 x m x 1) -> (n x m x m)
						lin_ids = tf.tile(tf.less(head_ids, dep_ids), [batch_size, 1, 1])
						# (n x 1 x m), (n x m x 1) -> (n x m x m)
						lin_xent = -tf.nn.softplus(tf.where(lin_ids, -lin_logits, lin_logits))
						# add the cross-entropy to the logits
						# (n x m x m), (n x m x m) -> (n x m x m)
						logits += tf.stop_gradient(lin_xent)
					if self.distance:
						# (n x m x m) - (1 x m x 1) -> (n x m x m)
						dist_targets = tf.abs(unlabeled_targets - dep_ids)
						# KL-divergence of the distance of each i,j pair
						# (1 x 1 x m) - (1 x m x 1) -> (n x m x m)
						dist_ids = tf.to_float(tf.tile(tf.abs(head_ids - dep_ids), [batch_size, 1, 1]))+1e-12
						# (n x m x m), (n x m x m) -> (n x m x m)
						#dist_kld = (dist_ids * tf.log(dist_lamda / dist_ids) + dist_ids - dist_lamda)
						dist_kld = -tf.log((dist_ids - dist_lamda)**2/2 + 1)
						# add the KL-divergence to the logits
						# (n x m x m), (n x m x m) -> (n x m x m)
						logits += tf.stop_gradient(dist_kld)
					#pdb.set_trace()
					#-----------------------------------------------------------
					# Note: here need a transpose as the target is the transpose graph(or opposite direction of adjacency graph)
					# (n x 2 x ma x mb) -> (n x 2 x mb x ma)
					if self.transposed:
						q_value=tf.transpose(q_value, [0,1,3,2])
					# Compute probabilities/cross entropy
					# (n x 2 x m x m) -> (n x m x m x 2)
					transposed_logits = tf.transpose(q_value, [0,2,3,1])
					probabilities=tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
					#TODO: what I want is still a probability of label 1, compared to the origin sigmoid(logits)? check later
					probabilities=probabilities[:,:,:,1]
					#label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
					# (n x m x m), (n x m x m x c), (n x m x m) -> ()
					# change sparse_softmax_cross_entropy to softmax_cross_entropy? It is the same in this situation
					loss = tf.losses.sparse_softmax_cross_entropy(unlabeled_targets, transposed_logits, weights=token_weights)
					#loss = tf.losses.sparse_sigmoid_cross_entropy(unlabeled_targets, transposed_logits, weights=token_weights)
					#pdb.set_trace()
					'''
					transposed_logits=tf.nn.softmax(transposed_logits,-1)
					L2_target=tf.cast(unlabeled_targets,dtype=tf.float32)
					loss = tf.reduce_sum((tf.pow(L2_target-transposed_logits[:,:,:,1],2)+tf.pow(1-L2_target-transposed_logits[:,:,:,0],2))\
							*tf.cast(token_weights,dtype=tf.float32))/tf.cast(batch_size,dtype=tf.float32)
					if debug:
						outputs['printdata']['L2_target']=L2_target
						outputs['printdata']['transposed_logits']=transposed_logits
					#'''
					if self.combine_loss:
						print('use combined loss')
						unary_probs=-unary
						# similar to q_value
						if self.transposed:
							unary_probs=tf.transpose(unary_probs,[0,3,2,1])
						unary_probs=tf.transpose(unary_probs, [0,1,3,2])
						#loss /= 2
						loss = loss*self.loss_weight + tf.losses.sparse_softmax_cross_entropy(unlabeled_targets, unary_probs, weights=token_weights)*self.loss_weight_unary
					'''
					# (n x m x m) -> (n x m x m)
					probabilities = tf.nn.sigmoid(logits) * tf.to_float(token_weights)#token weights is sentence length?
					# (n x m x m), (n x m x m), (n x m x m) -> ()
					loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=token_weights)#here label_smoothing is 0, the sigmoid XE have any effect?
					'''
					n_tokens = tf.to_float(tf.reduce_sum(token_weights))
					if self.linearize:
						lin_target_xent = lin_xent * unlabeled_targets
						loss -= tf.reduce_sum(lin_target_xent * tf.to_float(token_weights)) / (n_tokens + 1e-12)
					if self.distance:
						dist_target_kld = dist_kld * unlabeled_targets
						loss -= tf.reduce_sum(dist_target_kld * tf.to_float(token_weights)) / (n_tokens + 1e-12)
					
					#-----------------------------------------------------------
					# Compute predictions/accuracy 
					# precision/recall
					# (n x m x m) -> (n x m x m)
					#predictions = nn.greater(logits, 0, dtype=tf.int32) * token_weights#edge that predicted
					predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32) * token_weights
					# if self.compare_precision:
					# 	cond = tf.equal(transposed_logits[:,:,:,1], tf.expand_dims(tf.reduce_max(transposed_logits[:,:,:,1],-1),-1))
					# 	predictions = tf.where(cond, tf.cast(cond,tf.float32), tf.zeros_like(transposed_logits[:,:,:,1])) 
					# 	predictions = tf.cast(predictions,tf.int32) * token_weights
						
					# (n x m x m) (*) (n x m x m) -> (n x m x m)
					true_positives = predictions * unlabeled_targets
					# (n x m x m) -> ()
					n_predictions = tf.reduce_sum(predictions)
					n_targets = tf.reduce_sum(unlabeled_targets)
					n_true_positives = tf.reduce_sum(true_positives)
					# () - () -> ()
					n_false_positives = n_predictions - n_true_positives
					n_false_negatives = n_targets - n_true_positives
					# (n x m x m) -> (n)
					n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
					n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
					# (n) x 2 -> ()
					n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
			#-----------------------------------------------------------
			# Populate the output dictionary
			if debug:
				outputs['printdata']['logits']=transposed_logits
				outputs['printdata']['loss']=loss
			#print(123)
			outputs['unlabeled_targets'] = unlabeled_targets
			outputs['probabilities'] = probabilities
			outputs['unlabeled_loss'] = loss
			outputs['loss'] = loss
			
			outputs['unlabeled_predictions'] = predictions
			outputs['n_unlabeled_true_positives'] = n_true_positives
			outputs['n_unlabeled_false_positives'] = n_false_positives
			outputs['n_unlabeled_false_negatives'] = n_false_negatives
			outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
			outputs['predictions'] = predictions
			outputs['n_true_positives'] = n_true_positives
			outputs['n_false_positives'] = n_false_positives
			outputs['n_false_negatives'] = n_false_negatives
			outputs['n_correct_sequences'] = n_correct_sequences
			return outputs
	
	
	#=============================================================
	# token should be: 1:rel|2:acl|5:dep or 1|2|5
	def index(self, token):
		""""""
		
		nodes = []
		if token != '_':
			token = token.split('|')
			for edge in token:
				head = edge.split(':')[0]
				nodes.append(int(head))
		return nodes
	
	#=============================================================
	# index should be [1, 2, 5]
	def token(self, index):
		""""""
		
		return [str(head) for head in index]
	
	#=============================================================
	def get_root(self):
		""""""
		
		return '_'
	
	#=============================================================
	def __getitem__(self, key):
		if isinstance(key, six.string_types):
			nodes = []
			if key != '_':
				token = key.split('|')
				for edge in token:
					head = edge.split(':')[0]
					nodes.append(int(head))
			return nodes
		elif hasattr(key, '__iter__'):
			if len(key) > 0:
				if isinstance(key[0], six.integer_types + (np.int32, np.int64)):
					return '|'.join([str(head) for head in key])
				else:
					return [self[k] for k in key]
			else:
				return '_'
		else:
			raise ValueError('Key to GraphIndexVocab.__getitem__ must be (iterable of) strings or iterable of integers')
	
#***************************************************************

class SecondOrderGraphLBPVocab(GraphSecondLBPVocab, cv.SemheadVocab):
	pass
