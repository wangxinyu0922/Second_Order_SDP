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

from .second_order_vocab import SecondOrderVocab

#***************************************************************
class GraphSecondIndexAllVocab(SecondOrderVocab): #second order trilinear classifier
	""""""
	
	_depth = -1
	
	#=============================================================
	def __init__(self, *args, **kwargs):
		""""""
		
		kwargs['placeholder_shape'] = [None, None, None]
		super(GraphSecondIndexAllVocab, self).__init__(*args, **kwargs)
		return
	
	#=============================================================
	def get_bilinear_discriminator(self, layer, token_weights, variable_scope=None, reuse=False, debug=False, token_weights4D=None,prev_output=None):
		""""""
		#in fact here is get_trilinear_discriminator
		#pdb.set_trace()
		print("creating full mean-field graph")
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
				if not reuse:
					print('init: ',self.tri_std, self.tri_std_unary)
				# if self.as_score:
				unary = classifiers.bilinear_discriminator(
					unary_layer1, unary_layer2,
					hidden_keep_prob=hidden_keep_prob,
					add_linear=add_linear,tri_std=self.tri_std_unary)
				# else:
				# unary = classifiers.bilinear_classifier(
				# 	unary_layer1, unary_layer2, 2,
				# 	hidden_keep_prob=hidden_keep_prob,
				# 	add_linear=add_linear,tri_std=self.tri_std_unary)
				# head dep dep
				if self.use_sib:
					with tf.variable_scope('Sibling_11'):
						layer_sib_11 = classifiers.trilinear_discriminator_outer(
							sib_head, sib_dep, sib_dep,
							hidden_keep_prob=hidden_keep_prob_tri,
							add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
					with tf.variable_scope('Sibling_10'):
						layer_sib_10 = classifiers.trilinear_discriminator_outer(
							sib_head, sib_dep, sib_dep,
							hidden_keep_prob=hidden_keep_prob_tri,
							add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
				# head head+dep dep
				if self.use_gp:
					with tf.variable_scope('GrandParents_11'):
						layer_gp_11 = classifiers.trilinear_discriminator_outer(
							gp_head, gp_headdep, gp_dep,
							hidden_keep_prob=hidden_keep_prob_tri,
							add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
					with tf.variable_scope('GrandParents_10'):
						layer_gp_10 = classifiers.trilinear_discriminator_outer(
							gp_head, gp_headdep, gp_dep,
							hidden_keep_prob=hidden_keep_prob_tri,
							add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
					with tf.variable_scope('GrandParents_01'):
						layer_gp_01 = classifiers.trilinear_discriminator_outer(
							gp_head, gp_headdep, gp_dep,
							hidden_keep_prob=hidden_keep_prob_tri,
							add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
				# head dep head
				if self.use_cop:
					with tf.variable_scope('CoParents_11'):
						layer_cop_11 = classifiers.trilinear_discriminator_outer(
							cop_head, cop_dep, cop_head,
							hidden_keep_prob=hidden_keep_prob_tri,
							add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
					with tf.variable_scope('CoParents_10'):
						layer_cop_10 = classifiers.trilinear_discriminator_outer(
							cop_head, cop_dep, cop_head,
							hidden_keep_prob=hidden_keep_prob_tri,
							add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
						
				if debug:
					outputs['printdata']={}
					outputs['printdata']['q_value_orig']=-unary
					# if self.new_potential:
					# 	#pdb.set_trace()
					# 	if self.use_sib:
					# 		outputs['printdata']['layer_sib_old']=layer_sib
					# 	if self.use_cop:
					# 		outputs['printdata']['layer_cop_old']=layer_cop
					# 	if self.use_gp:
					# 		outputs['printdata']['layer_gp_old']=layer_gp
					# 	#outputs['printdata']['layer1']=layer1

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
					if self.layer_mask:
						print('use layer mask')
						unary=unary*tf.cast(tf.transpose(token_weights,[0,2,1]),dtype=tf.float32)
						if self.use_sib:
							layer_sib_11=layer_sib_11*token_weights4D
							layer_sib_10=layer_sib_10*token_weights4D
							if self.remove_root_child:
								# abc -> ab,ac
								layer_sib_11=layer_sib_11*self.token_weights_sib
								layer_sib_10=layer_sib_10*self.token_weights_sib
						if self.use_cop:
							layer_cop_11=layer_cop_11*token_weights4D
							layer_cop_10=layer_cop_10*token_weights4D
							if self.remove_root_child:
								#abc -> ab,cb
								layer_cop_11=layer_cop_11*self.token_weights_cop
								layer_cop_10=layer_cop_10*self.token_weights_cop
						if self.use_gp:
							layer_gp_11=layer_gp_11*token_weights4D
							layer_gp_10=layer_gp_10*token_weights4D
							layer_gp_01=layer_gp_01*token_weights4D
							if self.remove_root_child:
								#abc -> ab, bc
								layer_gp_11=layer_gp_11*self.token_weights_gp
								layer_gp_10=layer_gp_10*self.token_weights_gp
								layer_gp_01=layer_gp_01*self.token_weights_gp
					unary_potential=tf.stack([tf.zeros_like(unary),unary],axis=2)
					q_value=unary_potential#suppose 1 is label 1


					#1 sibling (n x ma x mb x mc) * (n x ma x mc) -> (n x ma x mb)
					#2 grand parent (n x ma x mb x mc) * (n x mb x mc) -> (n x ma x mb)
					#3 coparent (n x ma x mb x mc) * (n x mc x mb) -> (n x ma x mb)
					if self.new_potential:
						if self.use_sib:
							layer_sib_11 = layer_sib_11-tf.linalg.band_part(layer_sib_11,-1,0) + tf.transpose(tf.linalg.band_part(layer_sib_11,0,-1),perm=[0,1,3,2])
							layer_sib_10 = layer_sib_10-tf.linalg.band_part(layer_sib_10,-1,0) + tf.transpose(tf.linalg.band_part(layer_sib_10,0,-1),perm=[0,1,3,2])
							# (n x ma x mb x mc) -> (n x ma x mc x mb), means that changing ab=1,ac=0 to ab=0,ac=1
							layer_sib_01 = tf.transpose(layer_sib_10,perm=[0,1,3,2])
						if self.use_gp:
							layer_gp2_11 = tf.transpose(layer_gp_11,perm=[0,2,3,1])
							layer_gp2_10 = tf.transpose(layer_gp_10,perm=[0,2,3,1])
							layer_gp2_01 = tf.transpose(layer_gp_01,perm=[0,2,3,1])
						if self.use_cop:
							#(n x ma x mb x mc) -> (n x mb x ma x mc) 
							#in order to create a symmtric tensor on ma and mc
							layer_cop_11 = tf.transpose(layer_cop_11,perm=[0,2,1,3])
							layer_cop_10 = tf.transpose(layer_cop_10,perm=[0,2,1,3])
							# first set lower triangle part to be zero, then assign the upper triangle part transposed to lower triangle part
							layer_cop_11 = layer_cop_11 - tf.linalg.band_part(layer_cop_11,-1,0) + tf.transpose(tf.linalg.band_part(layer_cop_11,0,-1),perm=[0,1,3,2])
							layer_cop_10 = layer_cop_10 - tf.linalg.band_part(layer_cop_10,-1,0) + tf.transpose(tf.linalg.band_part(layer_cop_10,0,-1),perm=[0,1,3,2])
							# Finally (n x mb x ma x mc) -> (n x ma x mb x mc)
							layer_cop_11 = tf.transpose(layer_cop_11,perm=[0,2,1,3])
							layer_cop_10 = tf.transpose(layer_cop_10,perm=[0,2,1,3])
							# (n x ma x mb x mc) -> (n x mc x mb x ma), means that changing ab=1,ca=0 to ab=0,ca=1
							layer_cop_01 = tf.transpose(layer_cop_10,perm=[0,2,1,3])
					#======================CRF-RNN==========================
					for i in range(int(self.num_iteration)):
						q_value=tf.nn.softmax(q_value,2)
						if debug and i==0:
							outputs['q_value_old']=q_value
						if debug:
							outputs['q_value'+str(i)]=q_value
						if self.use_sib:
							second_temp_sib_11 = tf.einsum('nac,nabc->nab', q_value[:,:,1,:], layer_sib_11)
							second_temp_sib_10 = tf.einsum('nac,nabc->nab', q_value[:,:,0,:], layer_sib_10)
							second_temp_sib_01 = tf.einsum('nac,nabc->nab', q_value[:,:,1,:], layer_sib_01)
						else:
							second_temp_sib_11=0
							second_temp_sib_10=0
							second_temp_sib_01=0
						if self.use_gp:
							#'''
							#a->b->c
							second_temp_gp_11 = tf.einsum('nbc,nabc->nab', q_value[:,:,1,:], layer_gp_11)
							second_temp_gp_10 = tf.einsum('nbc,nabc->nab', q_value[:,:,0,:], layer_gp_10)
							second_temp_gp_01 = tf.einsum('nbc,nabc->nab', q_value[:,:,1,:], layer_gp_01)

							second_temp_gp2_11 = tf.einsum('nca,nabc->nab', q_value[:,:,1,:], layer_gp2_11)
							second_temp_gp2_10 = tf.einsum('nca,nabc->nab', q_value[:,:,0,:], layer_gp2_10)
							second_temp_gp2_01 = tf.einsum('nca,nabc->nab', q_value[:,:,1,:], layer_gp2_01)
						else:
							second_temp_gp_11=0
							second_temp_gp_10=0
							second_temp_gp_01=0

							second_temp_gp2_11=0
							second_temp_gp2_10=0
							second_temp_gp2_01=0
						if self.use_cop:
							with tf.device('/device:GPU:2'):  
								second_temp_cop_11 = tf.einsum('ncb,nabc->nab', q_value[:,:,1,:], layer_cop_11)
								second_temp_cop_10 = tf.einsum('ncb,nabc->nab', q_value[:,:,0,:], layer_cop_10)
								second_temp_cop_01 = tf.einsum('ncb,nabc->nab', q_value[:,:,1,:], layer_cop_01)
						else:
							second_temp_cop_11=0
							second_temp_cop_10=0
							second_temp_cop_01=0
							#'''
						if self.self_minus:
							#'''
							if self.use_sib:
								#-----------------------------------------------------------
								# minus all a = b = c part
								#(n x ma x mb) -> (n x ma) -> (n x ma x 1) | (n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(a,a)*p(a,b,a) 
								diag_sib1_11 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_sib_11,perm=[0,2,1,3])),perm=[0,2,1])
								diag_sib1_10 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,0,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_sib_10,perm=[0,2,1,3])),perm=[0,2,1])
								diag_sib1_01 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_sib_01,perm=[0,2,1,3])),perm=[0,2,1])
								# (n x ma x mb x mc) -> (n x ma x mb)
								#Q(a,b)*p(a,b,b)
								diag_sib2_11 = q_value[:,:,1,:] * tf.linalg.diag_part(layer_sib_11)
								diag_sib2_10 = q_value[:,:,0,:] * tf.linalg.diag_part(layer_sib_10)
								diag_sib2_01 = q_value[:,:,1,:] * tf.linalg.diag_part(layer_sib_01)
								
								second_temp_sib_11 = second_temp_sib_11 - diag_sib1_11 - diag_sib2_11
								second_temp_sib_10 = second_temp_sib_10 - diag_sib1_10 - diag_sib2_10
								second_temp_sib_01 = second_temp_sib_01 - diag_sib1_01 - diag_sib2_01
							if self.use_gp:
								#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(b,a)*p(a,b,a)
								diag_gp1_11 = tf.transpose(q_value[:,:,1,:],perm=[0,2,1]) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp_11,perm=[0,2,1,3])),perm=[0,2,1])
								diag_gp1_10 = tf.transpose(q_value[:,:,0,:],perm=[0,2,1]) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp_10,perm=[0,2,1,3])),perm=[0,2,1])
								diag_gp1_01 = tf.transpose(q_value[:,:,1,:],perm=[0,2,1]) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp_01,perm=[0,2,1,3])),perm=[0,2,1])
								#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
								#Q(b,b)*p(a,b,b)
								diag_gp2_11 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),1) * tf.linalg.diag_part(layer_gp_11)
								diag_gp2_10 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,0,:]),1) * tf.linalg.diag_part(layer_gp_10)
								diag_gp2_01 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),1) * tf.linalg.diag_part(layer_gp_01)

								#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(a,a)*p(a,b,a)
								diag_gp21_11 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp2_11,perm=[0,2,1,3])),perm=[0,2,1])
								diag_gp21_10 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,0,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp2_10,perm=[0,2,1,3])),perm=[0,2,1])
								diag_gp21_01 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),-1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp2_01,perm=[0,2,1,3])),perm=[0,2,1])
								#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
								#Q(b,a)*p(a,b,b)
								diag_gp22_11 = tf.transpose(q_value[:,:,1,:],perm=[0,2,1]) * tf.linalg.diag_part(layer_gp2_11)
								diag_gp22_10 = tf.transpose(q_value[:,:,0,:],perm=[0,2,1]) * tf.linalg.diag_part(layer_gp2_10)
								diag_gp22_01 = tf.transpose(q_value[:,:,1,:],perm=[0,2,1]) * tf.linalg.diag_part(layer_gp2_01)

								second_temp_gp_11 = second_temp_gp_11 - diag_gp1_11 - diag_gp2_11
								second_temp_gp_10 = second_temp_gp_10 - diag_gp1_10 - diag_gp2_10
								second_temp_gp_01 = second_temp_gp_01 - diag_gp1_01 - diag_gp2_01
								#c->a->b
								second_temp_gp2_11 = second_temp_gp2_11 - diag_gp21_11 - diag_gp22_11
								second_temp_gp2_10 = second_temp_gp2_10 - diag_gp21_10 - diag_gp22_10
								second_temp_gp2_01 = second_temp_gp2_01 - diag_gp21_01 - diag_gp22_01
								#second_temp_gp=second_temp_gp+second_temp_gp2
							if self.use_cop:
								#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
								#Q(a,b)*p(a,b,a)
								diag_cop1_11 = q_value[:,:,1,:] * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop_11,perm=[0,2,1,3])),perm=[0,2,1])
								diag_cop1_10 = q_value[:,:,0,:] * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop_10,perm=[0,2,1,3])),perm=[0,2,1])
								diag_cop1_01 = q_value[:,:,1,:] * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop_01,perm=[0,2,1,3])),perm=[0,2,1])
								#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
								#Q(b,b)*p(a,b,b)
								diag_cop2_11 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),1) * tf.linalg.diag_part(layer_cop_11)
								diag_cop2_10 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,0,:]),1) * tf.linalg.diag_part(layer_cop_10)
								diag_cop2_01 = tf.expand_dims(tf.linalg.diag_part(q_value[:,:,1,:]),1) * tf.linalg.diag_part(layer_cop_01)
								
								second_temp_cop_11 = second_temp_cop_11 - diag_cop1_11 - diag_cop2_11
								second_temp_cop_10 = second_temp_cop_10 - diag_cop1_10 - diag_cop2_10
								second_temp_cop_01 = second_temp_cop_01 - diag_cop1_01 - diag_cop2_01
							#'''

						# if debug:
						# 	if self.use_sib:
						# 		outputs['printdata']['second_temp_sib'+str(i)+'after']=second_temp_sib
						# 		outputs['printdata']['diag_sib1'+str(i)]=diag_sib1
						# 		outputs['printdata']['diag_sib2'+str(i)]=diag_sib2
						# 	if self.use_gp:
						# 		#outputs['printdata']['second_temp_gp']=second_temp_gp
						# 		outputs['printdata']['second_temp_gp'+str(i)+'after']=second_temp_gp
						# 		outputs['printdata']['second_temp_gp2'+str(i)+'after']=second_temp_gp2
						# 	if self.use_cop:
						# 		outputs['printdata']['second_temp_cop'+str(i)+'after']=second_temp_cop
						#pdb.set_trace()
						second_temp_1 = second_temp_sib_11 + second_temp_gp_11 + second_temp_gp2_11 + second_temp_cop_11 + second_temp_sib_10 + second_temp_gp_10 + second_temp_gp2_10 + second_temp_cop_10
						second_temp_0 = second_temp_sib_01 + second_temp_gp_01 + second_temp_gp2_01 + second_temp_cop_01
						if self.remove_loop:
							#pdb.set_trace()
							second_temp_1 = second_temp_1 - tf.linalg.diag(tf.linalg.diag_part(second_temp_1))
							second_temp_0 = second_temp_0 - tf.linalg.diag(tf.linalg.diag_part(second_temp_0))
						'''
						if not self.sibling_only:
							second_temp = second_temp_sib + second_temp_gp + second_temp_cop
						elif self.use_sib:
							second_temp = second_temp_sib 
						'''
						#Second order potential update function
						second_temp_1=unary_potential[:,:,1,:] + second_temp_1
						second_temp_0=second_temp_0
						q_value=tf.stack([second_temp_0,second_temp_1],axis=2)
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
						# outputs['printdata']['q_value']=q_value
						# outputs['q_value'+str(i+1)]=tf.nn.softmax(q_value,2)
						# outputs['printdata']['unary']=unary
						# #outputs['printdata']['binary']=binary
						# outputs['printdata']['second_temp']=second_temp
						if self.use_sib:
							outputs['printdata']['layer_sib_11']=layer_sib_11
							outputs['printdata']['layer_sib_10']=layer_sib_10
						if self.use_gp:
							#outputs['printdata']['second_temp_gp']=second_temp_gp
							# outputs['printdata']['second_temp_gp']=second_temp_gp
							# outputs['printdata']['second_temp_gp2']=second_temp_gp2
							outputs['printdata']['layer_gp_11']=layer_gp_11
							outputs['printdata']['layer_gp_10']=layer_gp_10
							outputs['printdata']['layer_gp_01']=layer_gp_01
						if self.use_cop:
							outputs['printdata']['layer_cop_11']=layer_cop_11
							outputs['printdata']['layer_cop_10']=layer_cop_10
							outputs['printdata']['layer_cop_01']=layer_cop_01
						# 	outputs['printdata']['second_temp_cop']=second_temp_cop
						# outputs['printdata']['layer1']=unary_layer1
						# outputs['printdata']['layer2']=unary_layer2
						# if not self.separate_embed:
						# 	outputs['printdata']['layer3']=layer3
						# outputs['printdata']['targets']=unlabeled_targets
						# outputs['printdata']['token_weights']=token_weights
						# if self.sibling_only:
						# 	outputs['printdata']['binary_weights']=binary_weights 
						# 	outputs['printdata']['binary']=layer_sib
						# if self.new_potential:
						# 		#outputs['printdata']['layer_sib2']=layer_sib2
						# 		#outputs['printdata']['layer_gp2']=layer_gp2
						# 		#outputs['printdata']['layer_cop2']=layer_cop2
						# 		pass
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


class SecondOrderGraphIndexAllVocab(GraphSecondIndexAllVocab, cv.SemheadVocab):
	pass