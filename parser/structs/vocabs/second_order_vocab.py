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

import numpy as np
import tensorflow as tf

from .base_vocabs import BaseVocab 
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, classifiers
import pdb
#***************************************************************
class SecondOrderVocab(BaseVocab):
	""""""
	
	#=============================================================
	def __init__(self, *args, **kwargs):
		""""""
		
		super(SecondOrderVocab, self).__init__(*args, **kwargs)
		
		self.PAD_STR = '_'
		self.PAD_IDX = -1
		self.ROOT_STR = '0'
		self.ROOT_IDX = 0
		self.BOS_STR = '<bos>'
		self.BOS_IDX = 999998
		self.EOS_STR = '<eos>'
		self.EOS_IDX = 999999
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
	def get_bos(self):
		""""""
		
		return self.BOS_STR
	
	#=============================================================
	def get_eos(self):
		""""""
		
		return self.EOS_STR

	#=============================================================
	def get_bilinear_classifier(self, layer, token_weights, variable_scope=None, reuse=False, debug=False, token_weights4D=None,prev_output=None,sentence_mask=None):
		""""""
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
				if sentence_mask==None:
					sentence_mask=tf.expand_dims(tf.cast(tf.transpose(token_weights,[0,2,1])[:,0],dtype=tf.float32),-1)
				else:
					sentence_mask=tf.expand_dims(tf.cast(sentence_mask,dtype=tf.float32),-1)
				
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
					if not reuse:
						print('init: ',self.tri_std, self.tri_std_unary)
					if self.as_score:
						unary = classifiers.bilinear_discriminator(
							unary_layer1, unary_layer2,
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear,tri_std=self.tri_std_unary)
					else:
						unary = classifiers.bilinear_classifier(
							unary_layer1, unary_layer2, 2,
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear,tri_std=self.tri_std_unary)
					
					# head dep dep
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
				if debug:
					outputs['printdata']={}
					outputs['printdata']['q_value_orig']=unary
					if self.new_potential:
						#pdb.set_trace()
						if self.use_sib:
							outputs['printdata']['layer_sib_old']=layer_sib
						if self.use_cop:
							outputs['printdata']['layer_cop_old']=layer_cop
						if self.use_gp:
							outputs['printdata']['layer_gp_old']=layer_gp
						#outputs['printdata']['layer1']=layer1

				if self.two_gpu:
					print('two gpu training for MF')
					#pdb.set_trace()
					GPUID=1
				else:
					GPUID=0
				with tf.device('/device:GPU:'+str(GPUID)):  
					#-----------------------------------------------------------
					#Let's start mean field algorithm and CRF-RNN here
					#normalize the probability of two labels (1 and 0)
					if prev_output!=None:
						label_layer=prev_output['label_layer']
						with tf.variable_scope('Label_trilinear'):
							label_sib,label_cop,label_gp=classifiers.trilinear_label_layer(label_layer, weight_type=1)
							layer_sib+=label_sib
							layer_cop+=label_cop
							layer_gp+=label_gp
					if self.layer_mask:
						print('use layer mask')
						if self.as_score:
							unary=unary*tf.cast(tf.transpose(token_weights,[0,2,1]),dtype=tf.float32)
						else:
							unary=unary*tf.cast(tf.expand_dims(tf.transpose(token_weights,[0,2,1]),-2),dtype=tf.float32)
						if self.use_sib:
							layer_sib=layer_sib*token_weights4D
							if self.remove_root_child:
								# abc -> ab,ac
								layer_sib=layer_sib*self.token_weights_sib
						if self.use_cop:
							layer_cop=layer_cop*token_weights4D
							if self.remove_root_child:
								#abc -> ab,cb
								layer_cop=layer_cop*self.token_weights_cop
						if self.use_gp:
							layer_gp=layer_gp*token_weights4D
							if self.remove_root_child:
								#abc -> ab, bc
								layer_gp=layer_gp*self.token_weights_gp
					if self.as_score:
						unary_potential=tf.stack([tf.zeros_like(unary),unary],axis=2)
					else:
						unary_potential=-unary
					q_value=unary_potential#suppose 1 is label 1

					#1 sibling (n x ma x mb x mc) * (n x ma x mc) -> (n x ma x mb)
					#2 grand parent (n x ma x mb x mc) * (n x mb x mc) -> (n x ma x mb)
					#3 coparent (n x ma x mb x mc) * (n x mc x mb) -> (n x ma x mb)
					if self.new_potential:
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
					#======================CRF-RNN==========================
					for i in range(int(self.num_iteration)):
						q_value=tf.nn.softmax(q_value,2)
						if debug and i==0:
							outputs['q_value_old']=q_value
						if debug:
							outputs['q_value'+str(i)]=q_value
						if self.use_sib:
							second_temp_sib = tf.einsum('nac,nabc->nab', q_value[:,:,1,:], layer_sib)
						else:
							second_temp_sib=0
						if self.use_gp:
							#'''
							#a->b->c
							second_temp_gp = tf.einsum('nbc,nabc->nab', q_value[:,:,1,:], layer_gp)
							second_temp_gp2 = tf.einsum('nca,nabc->nab', q_value[:,:,1,:], layer_gp2)
						else:
							second_temp_gp=0
							second_temp_gp2=0
						if self.use_cop:
							with tf.device('/device:GPU:2'):  
								second_temp_cop = tf.einsum('ncb,nabc->nab', q_value[:,:,1,:], layer_cop)
						else:
							second_temp_cop=0
							#'''
						if self.self_minus:
							#'''
							if self.use_sib:
								#-----------------------------------------------------------
								# minus all a = b = c part
								#(n x ma x mb) -> (n x ma) -> (n x ma x 1) | (n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(a,a)*p(a,b,a) 
								diag_sib1 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_sib,perm=[0,2,1,3])),perm=[0,2,1])
								# (n x ma x mb x mc) -> (n x ma x mb)
								#Q(a,b)*p(a,b,b)
								diag_sib2 = q_value[:,:,1,:] * tf.linalg.diag_part(layer_sib)
								
								second_temp_sib = second_temp_sib - diag_sib1 - diag_sib2
							if self.use_gp:
								#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(b,a)*p(a,b,a)
								diag_gp1 = tf.transpose(q_value[:,:,1,:],perm=[0,2,1]) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp,perm=[0,2,1,3])),perm=[0,2,1])
								#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
								#Q(b,b)*p(a,b,b)
								diag_gp2 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),1) * tf.linalg.diag_part(layer_gp)

								#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(a,a)*p(a,b,a)
								diag_gp21 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp2,perm=[0,2,1,3])),perm=[0,2,1])
								#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
								#Q(b,a)*p(a,b,b)
								diag_gp22 = tf.transpose(q_value[:,:,1,:],perm=[0,2,1]) * tf.linalg.diag_part(layer_gp2)

								if debug:
									#pdb.set_trace()
									'''
									outputs['binary_old']=layer_gp
									outputs['q_value_old']=q_value
									outputs['second_temp_sib_old']=second_temp_gp
									outputs['diag_sib1']=diag_gp1
									outputs['diag_sib2']=diag_gp2
									#'''
									pass
								second_temp_gp = second_temp_gp - diag_gp1 - diag_gp2
								#c->a->b
								second_temp_gp2 = second_temp_gp2 - diag_gp21 - diag_gp22
								#second_temp_gp=second_temp_gp+second_temp_gp2
							if self.use_cop:
								with tf.device('/device:GPU:2'):  
									#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
									#Q(a,b)*p(a,b,a)
									diag_cop1 = q_value[:,:,1,:] * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop,perm=[0,2,1,3])),perm=[0,2,1])
									#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
									#Q(b,b)*p(a,b,b)
									diag_cop2 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),1) * tf.linalg.diag_part(layer_cop)
								
								second_temp_cop = second_temp_cop - diag_cop1 - diag_cop2
							#'''

						if debug:
							if self.use_sib:
								outputs['printdata']['second_temp_sib'+str(i)+'after']=second_temp_sib
								outputs['printdata']['diag_sib1'+str(i)]=diag_sib1
								outputs['printdata']['diag_sib2'+str(i)]=diag_sib2
							if self.use_gp:
								#outputs['printdata']['second_temp_gp']=second_temp_gp
								outputs['printdata']['second_temp_gp'+str(i)+'after']=second_temp_gp
								outputs['printdata']['second_temp_gp2'+str(i)+'after']=second_temp_gp2
							if self.use_cop:
								outputs['printdata']['second_temp_cop'+str(i)+'after']=second_temp_cop
						#pdb.set_trace()
						second_temp = second_temp_sib + second_temp_gp + second_temp_gp2 + second_temp_cop
						if self.remove_loop:
							#pdb.set_trace()
							second_temp = second_temp - tf.linalg.diag(tf.linalg.diag_part(second_temp))
						'''
						if not self.sibling_only:
							second_temp = second_temp_sib + second_temp_gp + second_temp_cop
						elif self.use_sib:
							second_temp = second_temp_sib 
						'''
						#Second order potential update function
						if debug:
							outputs['printdata']['second_temp'+str(i)]=second_temp
						if self.as_score:
							second_temp=self.unary_weight * unary_potential[:,:,1,:] + second_temp
						else:
							second_temp=self.unary_weight * unary_potential[:,:,1,:] - second_temp
						q_value=tf.stack([unary_potential[:,:,0,:],second_temp],axis=2)
						if debug:
							outputs['printdata']['q_value'+str(i)]=q_value
					#q_value=-unary
					#CRF-RNN end
					#======================CRF-RNN==========================

					
					#-----------------------------------------------------------
					# Process the targets
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
					if debug:
						#outputs['printdata']['logits']=logits
						outputs['printdata']['q_value']=q_value
						outputs['q_value'+str(i+1)]=tf.nn.softmax(q_value,2)
						outputs['printdata']['unary']=unary
						#outputs['printdata']['binary']=binary
						outputs['printdata']['second_temp']=second_temp
						if self.use_sib:
							outputs['printdata']['second_temp_sib']=second_temp_sib
							outputs['printdata']['layer_sib']=layer_sib
						if self.use_gp:
							#outputs['printdata']['second_temp_gp']=second_temp_gp
							outputs['printdata']['second_temp_gp']=second_temp_gp
							outputs['printdata']['second_temp_gp2']=second_temp_gp2
							outputs['printdata']['layer_gp']=layer_gp
						if self.use_cop:
							outputs['printdata']['layer_cop']=layer_cop
							outputs['printdata']['second_temp_cop']=second_temp_cop
						outputs['printdata']['layer1']=unary_layer1
						outputs['printdata']['layer2']=unary_layer2
						if not self.separate_embed:
							outputs['printdata']['layer3']=layer3
						outputs['printdata']['targets']=unlabeled_targets
						outputs['printdata']['token_weights']=token_weights
						if self.sibling_only:
							outputs['printdata']['binary_weights']=binary_weights 
							outputs['printdata']['binary']=layer_sib
						if self.new_potential:
								#outputs['printdata']['layer_sib2']=layer_sib2
								#outputs['printdata']['layer_gp2']=layer_gp2
								#outputs['printdata']['layer_cop2']=layer_cop2
								pass
						'''
						outputs['printdata']['binary_weights']=binary_weights
						outputs['printdata']['binary_weights_cop']=binary_weights_cop
						outputs['printdata']['binary_weights_gp']=binary_weights_gp
						outputs['printdata']['unary_weights']=unary_weights
						#'''

					#-----------------------------------------------------------
					# Note: here need a transpose as the target is the transpose graph(or opposite direction of adjacency graph)
					# (n x ma x 2 x mb) -> (n x mb x 2 x ma)
					if self.transposed:
						q_value=tf.transpose(q_value, [0,3,2,1])
					# Compute probabilities/cross entropy
					# (n x m x 2 x m) -> (n x m x m x 2)
					transposed_logits = tf.transpose(q_value, [0,1,3,2])
					probabilities=tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
					#probabilities=tf.nn.sigmoid(transposed_logits[:,:,:,1])*token_weights
					#TODO: what I want is still a probability of label 1, compared to the origin sigmoid(logits)? check later
					probabilities=probabilities[:,:,:,1]
					#label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
					# (n x m), (n x m x m), (n x m) -> ()
					# pdb.set_trace()
					targets=unlabeled_targets
					logits=transposed_logits[:,:,:,1]
					loss = tf.losses.sparse_softmax_cross_entropy(targets,logits,weights=sentence_mask)
					# (n x m) -> (n x m x m x 1)
					one_hot_targets = tf.expand_dims(tf.one_hot(targets, bucket_size), -1)
					# (n x m) -> ()
					n_tokens = tf.to_float(tf.reduce_sum(token_weights))
					if self.linearize:
						# (n x m x m) -> (n x m x 1 x m)
						lin_xent_reshaped = tf.expand_dims(lin_xent, -2)
						# (n x m x 1 x m) * (n x m x m x 1) -> (n x m x 1 x 1)
						lin_target_xent = tf.matmul(lin_xent_reshaped, one_hot_targets)
						# (n x m x 1 x 1) -> (n x m)
						lin_target_xent = tf.squeeze(lin_target_xent, [-1, -2])
						# (n x m), (n x m), (n x m) -> ()
						loss -= tf.reduce_sum(lin_target_xent*tf.to_float(token_weights)) / (n_tokens + 1e-12)
					if self.distance:
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
					#pdb.set_trace()
					predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
					# (n x m) (*) (n x m) -> (n x m)
					sentence_mask=tf.cast(sentence_mask,dtype=tf.int32)[:,:,0]
					correct_tokens = nn.equal(targets, predictions) * sentence_mask
					# (n x m) -> (n)
					tokens_per_sequence = tf.reduce_sum(sentence_mask, axis=-1)
					# (n x m) -> (n)
					correct_tokens_per_sequence = tf.reduce_sum(correct_tokens, axis=-1)
					# (n), (n) -> (n)
					correct_sequences = nn.equal(tokens_per_sequence, correct_tokens_per_sequence)
			
					if self.linearize:
						lin_target_xent = lin_xent * unlabeled_targets
						loss -= tf.reduce_sum(lin_target_xent * tf.to_float(sentence_mask)) / (n_tokens + 1e-12)
					if self.distance:
						dist_target_kld = dist_kld * unlabeled_targets
						loss -= tf.reduce_sum(dist_target_kld * tf.to_float(sentence_mask)) / (n_tokens + 1e-12)
			
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
	def direction(self):
		try:
			return self._config.getboolean(self, 'direction')
		except:
			return False
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
	def extra_cop(self):
		try:
			return self._config.getboolean(self,'extra_cop')
		except:
			return False
	@property
	def transposed(self):
		return self._config.getboolean(self,'transposed')
	@property
	def unary_weight(self):
		try:
			return self._config.getfloat(self,'unary_weight')
		except:
			if self._config.getboolean(self,'unary_weight'):
				return int(self.use_cop)+int(self.use_sib)+2*int(self.use_gp)
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
	def iteration_loss(self):
		try:
			return self._config.getboolean(self,'iteration_loss')
		except:
			return False
	@property
	def iteration_loss_weight(self):
		try:
			return self._config.getfloat(self,'iteration_loss_weight')
		except:
			return 0.5
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
			return 0.5
	@property
	def tri_std_unary(self):
		try:
			return self._config.getfloat(self,'tri_std_unary')
		except:
			return 0.5
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
	def as_score(self):
		try:
			return self._config.getboolean(self,'as_score')
		except:
			return True
	@property
	def remove_root_child(self):
		try:
			return self._config.getboolean(self,'remove_root_child')
		except:
			return False
	@property
	def hidden_k(self):
		try:
			return self._config.getint(self,'hidden_k')
		except:
			return 200
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
class GraphSecondIndexVocab(SecondOrderVocab): #second order trilinear classifier
	""""""
	
	_depth = -1
	
	#=============================================================
	def __init__(self, *args, **kwargs):
		""""""
		kwargs['placeholder_shape'] = [None, None, None]
		super(GraphSecondIndexVocab, self).__init__(*args, **kwargs)
		return
	

	#=============================================================
	def get_bilinear_discriminator(self, layer, token_weights, variable_scope=None, reuse=False, debug=False, token_weights4D=None,prev_output=None):
		""""""
		#in fact here is get_trilinear_discriminator
		#pdb.set_trace()
		
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
					if not reuse:
						print('init: ',self.tri_std, self.tri_std_unary)
					if self.as_score:
						unary = classifiers.bilinear_discriminator(
							unary_layer1, unary_layer2,
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear,tri_std=self.tri_std_unary)
					else:
						unary = classifiers.bilinear_classifier(
							unary_layer1, unary_layer2, 2,
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear,tri_std=self.tri_std_unary)
					if self.separate_embed:
						print('separate')
						if self.test_new_potential:
							print('testing new potential')
							# head dep dep
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
							# head dep dep
							if self.use_sib:
								with tf.variable_scope('Sibling'):
									'''
									layer_sib, binary_weights = classifiers.trilinear_discriminator_new(
										sib_head, sib_dep, sib_dep,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear)
									'''
									layer_sib = classifiers.trilinear_discriminator_new(
										sib_head, sib_dep, sib_dep,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear, tri_std=self.tri_std)
									#'''
							# head head+dep dep
							if self.use_gp:
								with tf.variable_scope('GrandParents'):
									'''
									layer_gp, binary_weights_gp = classifiers.trilinear_discriminator_new(
										gp_head, gp_headdep, gp_dep,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear)
									'''
									layer_gp = classifiers.trilinear_discriminator_new(
										gp_head, gp_headdep, gp_dep,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear, tri_std=self.tri_std)
									#'''
							# head dep head
							if self.use_cop:
								with tf.variable_scope('CoParents'):
									'''
									layer_cop, binary_weights_cop = classifiers.trilinear_discriminator_new(
										cop_head, cop_dep, cop_head,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear)
									'''
									layer_cop = classifiers.trilinear_discriminator_new(
										cop_head, cop_dep, cop_head,
										hidden_keep_prob=hidden_keep_prob_tri,
										add_linear=add_linear, tri_std=self.tri_std)
									#'''
					elif not self.old_trilin:
						print('head dep')
						# head dep dep
						if self.use_sib:
							with tf.variable_scope('Sibling'):
								layer_sib = classifiers.trilinear_discriminator_new(
									layer1, layer2, layer2,
									hidden_keep_prob=hidden_keep_prob_tri,
									add_linear=add_linear, tri_std=self.tri_std)
						# head head+dep dep
						if self.use_gp:
							with tf.variable_scope('GrandParents'):
								layer_gp = classifiers.trilinear_discriminator_new(
									layer1, layer3, layer2,
									hidden_keep_prob=hidden_keep_prob_tri,
									add_linear=add_linear, tri_std=self.tri_std)
						# head dep head
						if self.use_cop:
							with tf.variable_scope('CoParents'):
								layer_cop = classifiers.trilinear_discriminator_new(
									layer1, layer2, layer1,
									hidden_keep_prob=hidden_keep_prob_tri,
									add_linear=add_linear, tri_std=self.tri_std)
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
				if debug:
					outputs['printdata']={}
					outputs['printdata']['q_value_orig']=unary
					if self.new_potential:
						#pdb.set_trace()
						if self.use_sib:
							outputs['printdata']['layer_sib_old']=layer_sib
						if self.use_cop:
							outputs['printdata']['layer_cop_old']=layer_cop
						if self.use_gp:
							outputs['printdata']['layer_gp_old']=layer_gp
						#outputs['printdata']['layer1']=layer1

				if self.two_gpu:
					print('two gpu training for MF')
					#pdb.set_trace()
					GPUID=1
				else:
					GPUID=0
				with tf.device('/device:GPU:'+str(GPUID)):  
					#-----------------------------------------------------------
					#Let's start mean field algorithm and CRF-RNN here
					#normalize the probability of two labels (1 and 0)
					if prev_output!=None:
						label_layer=prev_output['label_layer']
						with tf.variable_scope('Label_trilinear'):
							label_sib,label_cop,label_gp=classifiers.trilinear_label_layer(label_layer, weight_type=1)
							layer_sib+=label_sib
							layer_cop+=label_cop
							layer_gp+=label_gp
					if self.layer_mask:
						print('use layer mask')
						if self.as_score:
							unary=unary*tf.cast(tf.transpose(token_weights,[0,2,1]),dtype=tf.float32)
						else:
							unary=unary*tf.cast(tf.expand_dims(tf.transpose(token_weights,[0,2,1]),-2),dtype=tf.float32)
						if self.use_sib:
							layer_sib=layer_sib*token_weights4D
							if self.remove_root_child:
								# abc -> ab,ac
								layer_sib=layer_sib*self.token_weights_sib
						if self.use_cop:
							layer_cop=layer_cop*token_weights4D
							if self.remove_root_child:
								#abc -> ab,cb
								layer_cop=layer_cop*self.token_weights_cop
						if self.use_gp:
							layer_gp=layer_gp*token_weights4D
							if self.remove_root_child:
								#abc -> ab, bc
								layer_gp=layer_gp*self.token_weights_gp
					if self.as_score:
						unary_potential=tf.stack([tf.zeros_like(unary),unary],axis=2)
					else:
						unary_potential=-unary
					q_value=unary_potential#suppose 1 is label 1

					#1 sibling (n x ma x mb x mc) * (n x ma x mc) -> (n x ma x mb)
					#2 grand parent (n x ma x mb x mc) * (n x mb x mc) -> (n x ma x mb)
					#3 coparent (n x ma x mb x mc) * (n x mc x mb) -> (n x ma x mb)
					if self.new_potential:
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
					#======================CRF-RNN==========================
					for i in range(int(self.num_iteration)):
						q_value=tf.nn.softmax(q_value,2)
						if debug and i==0:
							outputs['q_value_old']=q_value
						if debug:
							outputs['q_value'+str(i)]=q_value
						if self.use_sib:
							second_temp_sib = tf.einsum('nac,nabc->nab', q_value[:,:,1,:], layer_sib)
						else:
							second_temp_sib=0
						if self.use_gp:
							#'''
							#a->b->c
							second_temp_gp = tf.einsum('nbc,nabc->nab', q_value[:,:,1,:], layer_gp)
							second_temp_gp2 = tf.einsum('nca,nabc->nab', q_value[:,:,1,:], layer_gp2)
						else:
							second_temp_gp=0
							second_temp_gp2=0
						if self.use_cop:
							with tf.device('/device:GPU:2'):  
								second_temp_cop = tf.einsum('ncb,nabc->nab', q_value[:,:,1,:], layer_cop)
						else:
							second_temp_cop=0
							#'''
						if self.self_minus:
							#'''
							if self.use_sib:
								#-----------------------------------------------------------
								# minus all a = b = c part
								#(n x ma x mb) -> (n x ma) -> (n x ma x 1) | (n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(a,a)*p(a,b,a) 
								diag_sib1 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_sib,perm=[0,2,1,3])),perm=[0,2,1])
								# (n x ma x mb x mc) -> (n x ma x mb)
								#Q(a,b)*p(a,b,b)
								diag_sib2 = q_value[:,:,1,:] * tf.linalg.diag_part(layer_sib)
								if debug:
									outputs['printdata']['second_temp_sib'+str(i)+'before']=second_temp_sib
								second_temp_sib = second_temp_sib - diag_sib1 - diag_sib2
							if self.use_gp:
								#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(b,a)*p(a,b,a)
								diag_gp1 = tf.transpose(q_value[:,:,1,:],perm=[0,2,1]) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp,perm=[0,2,1,3])),perm=[0,2,1])
								#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
								#Q(b,b)*p(a,b,b)
								diag_gp2 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),1) * tf.linalg.diag_part(layer_gp)

								#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(a,a)*p(a,b,a)
								diag_gp21 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp2,perm=[0,2,1,3])),perm=[0,2,1])
								#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
								#Q(b,a)*p(a,b,b)
								diag_gp22 = tf.transpose(q_value[:,:,1,:],perm=[0,2,1]) * tf.linalg.diag_part(layer_gp2)

								if debug:
									#pdb.set_trace()
									'''
									outputs['binary_old']=layer_gp
									outputs['q_value_old']=q_value
									outputs['second_temp_sib_old']=second_temp_gp
									outputs['diag_sib1']=diag_gp1
									outputs['diag_sib2']=diag_gp2
									#'''
									pass
								second_temp_gp = second_temp_gp - diag_gp1 - diag_gp2
								#c->a->b
								second_temp_gp2 = second_temp_gp2 - diag_gp21 - diag_gp22
								#second_temp_gp=second_temp_gp+second_temp_gp2
							if self.use_cop:
								with tf.device('/device:GPU:2'):  
									#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
									#Q(a,b)*p(a,b,a)
									diag_cop1 = q_value[:,:,1,:] * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop,perm=[0,2,1,3])),perm=[0,2,1])
									#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
									#Q(b,b)*p(a,b,b)
									diag_cop2 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),1) * tf.linalg.diag_part(layer_cop)
								
								second_temp_cop = second_temp_cop - diag_cop1 - diag_cop2
							#'''

						if debug:
							if self.use_sib:
								outputs['printdata']['second_temp_sib'+str(i)+'after']=second_temp_sib
								outputs['printdata']['diag_sib1'+str(i)]=diag_sib1
								outputs['printdata']['diag_sib2'+str(i)]=diag_sib2
							if self.use_gp:
								#outputs['printdata']['second_temp_gp']=second_temp_gp
								outputs['printdata']['second_temp_gp'+str(i)+'after']=second_temp_gp
								outputs['printdata']['second_temp_gp2'+str(i)+'after']=second_temp_gp2
							if self.use_cop:
								outputs['printdata']['second_temp_cop'+str(i)+'after']=second_temp_cop
						#pdb.set_trace()
						second_temp = second_temp_sib + second_temp_gp + second_temp_gp2 + second_temp_cop
						if self.remove_loop:
							#pdb.set_trace()
							second_temp = second_temp - tf.linalg.diag(tf.linalg.diag_part(second_temp))
						'''
						if not self.sibling_only:
							second_temp = second_temp_sib + second_temp_gp + second_temp_cop
						elif self.use_sib:
							second_temp = second_temp_sib 
						'''
						#Second order potential update function
						if debug:
							outputs['printdata']['second_temp'+str(i)]=second_temp
						if self.as_score:
							second_temp=self.unary_weight * unary_potential[:,:,1,:] + second_temp
						else:
							second_temp=self.unary_weight * unary_potential[:,:,1,:] - second_temp
						q_value=tf.stack([unary_potential[:,:,0,:],second_temp],axis=2)
						if debug:
							outputs['printdata']['q_value'+str(i)]=q_value
					#q_value=-unary
					#CRF-RNN end
					#======================CRF-RNN==========================

					
					#-----------------------------------------------------------
					# Process the targets
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
					if debug:
						#outputs['printdata']['logits']=logits
						outputs['printdata']['q_value']=q_value
						outputs['q_value'+str(i+1)]=tf.nn.softmax(q_value,2)
						outputs['printdata']['unary']=unary
						#outputs['printdata']['binary']=binary
						outputs['printdata']['second_temp']=second_temp
						if self.use_sib:
							outputs['printdata']['second_temp_sib']=second_temp_sib
							outputs['printdata']['layer_sib']=layer_sib
						if self.use_gp:
							#outputs['printdata']['second_temp_gp']=second_temp_gp
							outputs['printdata']['second_temp_gp']=second_temp_gp
							outputs['printdata']['second_temp_gp2']=second_temp_gp2
							outputs['printdata']['layer_gp']=layer_gp
						if self.use_cop:
							outputs['printdata']['layer_cop']=layer_cop
							outputs['printdata']['second_temp_cop']=second_temp_cop
						outputs['printdata']['layer1']=unary_layer1
						outputs['printdata']['layer2']=unary_layer2
						if not self.separate_embed:
							outputs['printdata']['layer3']=layer3
						outputs['printdata']['targets']=unlabeled_targets
						outputs['printdata']['token_weights']=token_weights
						if self.sibling_only:
							outputs['printdata']['binary_weights']=binary_weights 
							outputs['printdata']['binary']=layer_sib
						if self.new_potential:
								#outputs['printdata']['layer_sib2']=layer_sib2
								#outputs['printdata']['layer_gp2']=layer_gp2
								#outputs['printdata']['layer_cop2']=layer_cop2
								pass
						'''
						outputs['printdata']['binary_weights']=binary_weights
						outputs['printdata']['binary_weights_cop']=binary_weights_cop
						outputs['printdata']['binary_weights_gp']=binary_weights_gp
						outputs['printdata']['unary_weights']=unary_weights
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
					# (n x ma x 2 x mb) -> (n x mb x 2 x ma)
					if self.transposed:
						q_value=tf.transpose(q_value, [0,3,2,1])
					# Compute probabilities/cross entropy
					# (n x m x 2 x m) -> (n x m x m x 2)
					transposed_logits = tf.transpose(q_value, [0,1,3,2])
					probabilities=tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
					#probabilities=tf.nn.sigmoid(transposed_logits[:,:,:,1])*token_weights
					#TODO: what I want is still a probability of label 1, compared to the origin sigmoid(logits)? check later
					probabilities=probabilities[:,:,:,1]
					#label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
					# (n x m x m), (n x m x m x c), (n x m x m) -> ()
					# change sparse_softmax_cross_entropy to softmax_cross_entropy? It is the same in this situation
					loss = tf.losses.sparse_softmax_cross_entropy(unlabeled_targets, transposed_logits, weights=token_weights)
					#loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, transposed_logits[:,:,:,1], weights=token_weights)
					#pdb.set_trace()
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
					#predictions = nn.greater(transposed_logits[:,:,:,1], 0, dtype=tf.int32) * token_weights#edge that predicted
					predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32) * token_weights
					# if self.compare_precision:
					# 	#pdb.set_trace()
					# 	# (n x m x m) -> (n x m)
					# 	#temp_predictions = tf.argmax(transposed_logits[:,:,:,1], axis=-1, output_type=tf.int32)
					# 	# (n x m) -> (n x m x m)
					# 	cond = tf.equal(transposed_logits[:,:,:,1], tf.expand_dims(tf.reduce_max(transposed_logits[:,:,:,1],-1),-1))
					# 	predictions = tf.where(cond, tf.cast(cond,tf.float32), tf.zeros_like(transposed_logits[:,:,:,1])) 
					# 	predictions = tf.cast(predictions,tf.int32) * token_weights
					# 	# # (n x m) (*) (n x m) -> (n x m)
					# 	# # (n x m) (*) (n x m) -> (n x m)
					# 	# n_true_positives_temp = tf.reduce_sum(nn.equal(tf.argmax(unlabeled_targets,axis=-1, output_type=tf.int32), temp_predictions) * self.token_weights)
					# 	# n_predictions_temp = tf.reduce_sum(predictions)
					# 	# n_false_positives_temp = n_predictions_temp - n_true_positives_temp
					# 	# if debug:
					# 	# 	outputs['printdata']['tp']=n_true_positives_temp
					# 	# 	outputs['printdata']['fp']=n_false_positives_temp
					# 	# 	outputs['printdata']['temp_predictions']=temp_predictions
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
		if prev_output!=None:
			if debug:
				outputs['printdata']['label_gp']=label_gp
				outputs['printdata']['label_cop']=label_cop
				outputs['printdata']['label_sib']=label_sib
				outputs['printdata']['label_layer']=label_layer
				outputs['printdata']['label_predictions']=prev_output['label_predictions']
			label_loss=prev_output['label_loss']
			predictions = prev_output['label_predictions']
			label_targets = prev_output['label_targets']
			label_probabilities = prev_output['label_probabilities']
			#-----------------------------------------------------------
			# Compute the predictions/accuracy
			# ---------------Unlabeled data----------------------------
			unlabeled_predictions = outputs['unlabeled_predictions']
			unlabeled_targets = outputs['unlabeled_targets']
			# (n x m x m) -> (n x m x m x 1)
			head_probabilities = tf.expand_dims(tf.stop_gradient(outputs['probabilities']), axis=-1)
			# ---------------Unlabeled data----------------------------
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
			rho = prev_output['rho']
			outputs['label_targets'] = label_targets
			outputs['probabilities'] = label_probabilities * head_probabilities
			outputs['label_loss'] = prev_output['label_loss']
			# Combination of labeled loss and unlabeled loss
			outputs['loss'] = 2*((1-rho) * outputs['loss'] + rho * label_loss)
			
			outputs['n_true_positives'] = n_true_positives
			outputs['n_false_positives'] = n_false_positives
			outputs['n_false_negatives'] = n_false_negatives
			outputs['n_correct_sequences'] = n_correct_sequences
			outputs['n_correct_label_tokens'] = n_correct_label_tokens
			outputs['n_correct_label_sequences'] = n_correct_label_sequences
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


class SecondOrderGraphIndexVocab(GraphSecondIndexVocab, cv.SemheadVocab):
	pass