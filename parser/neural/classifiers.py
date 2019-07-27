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

from . import nn
from . import nonlin
import pdb
#***************************************************************
def hidden(layer, hidden_size, hidden_func=nonlin.relu, hidden_keep_prob=1.):
	""""""

	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = tf.get_variable('Weights', shape=[input_size, hidden_size])#, initializer=tf.orthogonal_initializer)
	biases = tf.get_variable('Biases', shape=[hidden_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	layer = nn.reshape(layer, [-1, input_size])
	layer = tf.matmul(layer, weights) + biases
	layer = hidden_func(layer)
	layer = nn.reshape(layer, layer_shape + [hidden_size])
	return layer

#===============================================================
def hiddens(layer, hidden_sizes, hidden_func=nonlin.relu, hidden_keep_prob=1.):
	""""""
	#pdb.set_trace()
	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = []
	for i, hidden_size in enumerate(hidden_sizes):
		weights.append(tf.get_variable('Weights-%d' % i, shape=[input_size, hidden_size]))#, initializer=tf.orthogonal_initializer))
	weights = tf.concat(weights, axis=1)
	hidden_size = sum(hidden_sizes)
	biases = tf.get_variable('Biases', shape=[hidden_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	layer = nn.reshape(layer, [-1, input_size])
	layer = tf.matmul(layer, weights) + biases
	layer = hidden_func(layer)
	layer = nn.reshape(layer, layer_shape + [hidden_size])
	layers = tf.split(layer, hidden_sizes, axis=-1)
	return layers

#===============================================================
def linear_classifier(layer, output_size, hidden_keep_prob=1.):
	""""""
	
	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
	biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	# (n x m x d) -> (nm x d)
	layer_reshaped = nn.reshape(layer, [-1, input_size])
	
	# (nm x d) * (d x o) -> (nm x o)
	layer = tf.matmul(layer_reshaped, weights) + biases
	# (nm x o) -> (n x m x o)
	layer = nn.reshape(layer, layer_shape + [output_size])
	return layer
	
#===============================================================
def linear_attention(layer, hidden_keep_prob=1.):
	""""""
	
	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = tf.get_variable('Weights', shape=[input_size, 1], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	# (n x m x d) -> (nm x d)
	layer_reshaped = tf.reshape(layer, [-1, input_size])
	
	# (nm x d) * (d x 1) -> (nm x 1)
	attn = tf.matmul(layer_reshaped, weights)
	# (nm x 1) -> (n x m)
	attn = tf.reshape(attn, layer_shape)
	# (n x m) -> (n x m)
	attn = tf.nn.sigmoid(attn)
	# (n x m) -> (n x 1 x m)
	soft_attn = tf.expand_dims(attn, axis=-2)
	# (n x 1 x m) * (n x m x d) -> (n x 1 x d)
	weighted_layer = tf.matmul(soft_attn, layer)
	# (n x 1 x d) -> (n x d)
	weighted_layer = tf.squeeze(weighted_layer, -2)
	return attn, weighted_layer

#===============================================================
def deep_linear_attention(layer, hidden_size, hidden_func=tf.identity, hidden_keep_prob=1.):
	""""""
	
	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = tf.get_variable('Weights', shape=[input_size, hidden_size+1], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	# (n x m x d) -> (nm x d)
	layer_reshaped = tf.reshape(layer, [-1, input_size])
	
	# (nm x d) * (d x o+1) -> (nm x o+1)
	attn = tf.matmul(layer_reshaped, weights)
	# (nm x o+1) -> (nm x 1), (nm x o)
	attn, layer = tf.split(attn, [1, hidden_size], axis=-1)
	# (nm x 1) -> (nm x 1)
	attn = tf.nn.sigmoid(attn)
	# (nm x 1) o (nm x o) -> (nm x o)
	weighted_layer = hidden_func(layer) * attn
	# (nm x 1) -> (n x m)
	attn = tf.reshape(attn, layer_shape)
	# (nm x o) -> (n x m x o)
	weighted_layer = nn.reshape(weighted_layer, layer_shape+[hidden_size])
	return attn, weighted_layer
	
#===============================================================
def batch_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob, add_linear=True):
	""""""

	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+add_linear
	input2_size = layer2.get_shape().as_list()[-1]+add_linear
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-add_linear])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	if add_linear:
		ones = tf.ones(ones_shape)
		layer1 = tf.concat([layer1, ones], -1)
		layer2 = tf.concat([layer2, ones], -1)
		biases = 0
	else:
		biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
		# (o) -> (o x 1)
		biases = nn.reshape(biases, [output_size, 1])
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (nm x d x 1)
	layer2 = nn.reshape(layer2, [-1, input2_size, 1])
	# (d x o x d) -> (d x od)
	weights = nn.reshape(weights, [input1_size, output_size*input2_size])
	
	# (nm x d) * (d x od) -> (nm x od)
	layer = tf.matmul(layer1, weights)
	# (nm x od) -> (nm x o x d)
	layer = nn.reshape(layer, [-1, output_size, input2_size])
	# (nm x o x d) * (nm x d x 1) -> (nm x o x 1)
	layer = tf.matmul(layer, layer2)
	# (nm x o x 1) -> (n x m x o)
	layer = nn.reshape(layer, layer_shape + [output_size]) + biases
	return layer

#===============================================================
def bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True, target_model='CRF', tri_std=0.5):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+add_linear
	input2_size = layer2.get_shape().as_list()[-1]+add_linear
	ones_shape = tf.stack(layer_shape + [1])
	
	if target_model=='CRF':
		weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	elif target_model=='LBP':
		weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	#weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.random_normal_initializer(stddev=0.01))
	
	#weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.contrib.layers.xavier_initializer())
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-add_linear])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	if add_linear:#add linear means linear layer?
		ones = tf.ones(ones_shape)
		layer1 = tf.concat([layer1, ones], -1)
		layer2 = tf.concat([layer2, ones], -1)
		biases = 0
	else:
		biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
		# (o) -> (o x 1)
		biases = nn.reshape(biases, [output_size, 1])
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input2_size])
	# (d x o x d) -> (d x od)
	weights = nn.reshape(weights, [input1_size, output_size*input2_size])
	
	# (nm x d) * (d x od) -> (nm x od)
	layer = tf.matmul(layer1, weights)
	# (nm x od) -> (n x mo x d)
	layer = nn.reshape(layer, [-1, bucket_size*output_size, input2_size])
	# (n x mo x d) * (n x m x d) -> (n x mo x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x mo x m) -> (n x m x o x m)
	layer = nn.reshape(layer, layer_shape + [output_size, bucket_size]) + biases
	#return layer, weights
	return layer

#===============================================================
def trilinear_classifier(layer1, layer2, layer3, output_size, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x o^2 x d) 
	#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, output_size, input3_size], initializer=tf.truncated_normal_initializer())
	#(o^2 x d x d x d) 
	weights = tf.get_variable('trilinear_Weights', shape=[output_size**2, input1_size, input2_size, input3_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	if add_linear:
		ones = tf.ones(ones_shape)
		layer1 = tf.concat([layer1, ones], -1)
		layer2 = tf.concat([layer2, ones], -1)
		layer3 = tf.concat([layer3, ones], -1)
		biases = 0
	else:
		biases = tf.get_variable('Biases', shape=[output_size**2], initializer=tf.zeros_initializer)
		# (o) -> (o x 1)
		biases = nn.reshape(biases, [output_size**2, 1])
	#pdb.set_trace()
	# (n x m x d) -> (n x m x d)
	layer1 = nn.reshape(layer1, [-1, bucket_size, input1_size])
	# (n x m x d) -> (n x m x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (n x m x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	
	# (n x ma x d) * (o^2 x d x d x d) -> (n x o^2 x ma x d x d)
	layer = tf.einsum('oijk,abi->aobjk', weights, layer1)
	#layer = tf.tensordot(layer1, weights, axes=[[1], [1]]) 
	# (nm x o^2 x d x d) -> (n x m x o^2 x d x d)
	#layer = nn.reshape(layer, [-1, output_size, bucket_size, input2_size, input3_size])
	# (n x o^2 x ma x d x d) * (n x mc x d) -> (n x o^2 x ma x d x mc)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer = tf.einsum('aobij,acj->aobic', layer, layer3)
	# (n x o^2 x ma x d x mc) * (n x mb x d) -> (n x o^2 x ma x mb x mc)
	layer = tf.einsum('aobic,adi->aobdc', layer, layer2)
	layer += biases
	# layer = (n x o^2 x ma x mb x mc) -> (n x o x o x ma x mb x mc)
	layer = tf.reshape(layer, [-1, output_size, output_size] + [bucket_size]*3)
	#return layer
	return layer

#===============================================================
def diagonal_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True):
	""""""
	#here is token classifier
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
	
	if add_linear: #why here do not use weights2?
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer1 = nn.reshape(lin_layer1, layer_shape + [output_size])
		# (n x m x o) -> (n x m x o x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)
	
	# (n x m x d) -> (n x m x 1 x d)
	layer1 = nn.reshape(layer1, [-1, bucket_size, 1, input_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	# (d x o) -> (o x d)
	weights = tf.transpose(weights, [1, 0])
	# (o) -> (o x 1)
	biases = nn.reshape(biases, [output_size, 1])
	# means every word in layer1 have m label?
	# (n x m x 1 x d) (*) (o x d) -> (n x m x o x d)
	layer = layer1 * weights
	# (n x m x o x d) -> (n x mo x d)
	layer = nn.reshape(layer, [-1, bucket_size*output_size, input_size])
	# (n x mo x d) * (n x m x d) -> (n x mo x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x mo x m) -> (n x m x o x m)
	layer = nn.reshape(layer, layer_shape + [output_size, bucket_size])
	if add_linear:
		# (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
		layer += lin_layer1 + lin_layer2
	# (n x m x o x m) + (o x 1) -> (n x m x o x m)
	layer += biases
	return layer

#===============================================================
def diagonal_bilinear_layer(layer1, layer2, hidden_func=nonlin.relu, hidden_keep_prob=1., add_linear=True):
	""""""
	#here is token classifier
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.1))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.1))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	biases = tf.get_variable('Biases', shape=[input_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
	
	if add_linear:
		# (n x m x d) * (d x o) -> (n x m x o) -> # (n x m x o) -> (n x m x 1 x o)
		lin_layer1 = tf.expand_dims(tf.tensordot(layer1, weights1, axes=[[-1],[0]]),axis=-2)
		# (n x m x d) * (d x o) -> (n x m x o) -> # (n x m x o) -> (n x 1 x m x o)
		lin_layer2 = tf.expand_dims(tf.tensordot(layer2, weights2, axes=[[-1],[0]]),axis=-3)
	# (n x m x d) -> (n x m x d x 1)
	layer1 = tf.expand_dims(layer1,axis=-1)
	# (n x m x d x 1) * (d x o) -> (n x m x d x o)
	layer = layer1 * weights
	# (n x m x d x o) * (n x m x d) -> (n x m x m x o)
	layer = tf.einsum('nmdo,nad->nmao', layer, layer2)
	if add_linear:
		# (n x m x m x o) + (n x m x 1 x o) + (n x 1 x m x o) -> (n x m x m x o)
		layer += lin_layer1 + lin_layer2
	# (o) -> (1 x o)
	biases = nn.reshape(biases, [1,input_size])
	# (n x m x m x o) + (1 x o) -> (n x m x m x o)
	layer += biases
	# (n x m x m x d)
	#layer = hidden_func(layer)
	return layer





#===============================================================
def trilinear_discriminator_new(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True,target_model='CRF',tri_std=0.01):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	if target_model=='CRF':
		#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.truncated_normal_initializer())
		weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.random_normal_initializer(stddev=tri_std))
		
	elif target_model=='LBP':
		weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.truncated_normal_initializer())
	#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.contrib.layers.xavier_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	#pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (nm x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (nm x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	
	# (nm x d) * (d x d x d) -> (nm x d x d)
	layer = tf.tensordot(layer1, weights, axes=[[1], [0]]) 
	# (nm x d x d) -> (nm x d x d)
	layer = nn.reshape(layer, [-1, bucket_size, input2_size, input3_size])
	# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer = tf.einsum('abij,acj->abic', layer, layer3)
	# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
	layer = tf.einsum('abic,adi->abdc', layer, layer2)
	# (n x mo x m x m) -> (n x m x m x m)
	layer = nn.reshape(layer, layer_shape + [bucket_size]*2)
	# layer = (n x ma x mc x mb) 
	#return layer,weights
	return layer

#===============================================================
def trilinear_discriminator_outer(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True,target_model='CRF',tri_std=0.01, hidden_k=200):
	""""""
	#use outer tensor product for trilinear layer size of k x d x 1
	#pdb.set_trace()
	#print(tri_std)
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	if target_model=='CRF':
		weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
		weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
		weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
	elif target_model=='LBP':
		weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
		weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
		weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	#pdb.set_trace()
	# (n x m x d) * (d x k) -> (n x m x k)
	layer1_tmp = tf.tensordot(layer1, weights1, axes=[[-1], [0]])
	# (n x m x d) * (d x k) -> (n x m x k)
	layer2_tmp = tf.tensordot(layer2, weights2, axes=[[-1], [0]])
	# (n x m x d) * (d x k) -> (n x m x k)
	layer3_tmp = tf.tensordot(layer3, weights3, axes=[[-1], [0]])
	#(n x ma x k) * (n x mb x k) -> (n x ma x mb x k)
	layer12_tmp = tf.einsum('nak,nbk->nabk',layer1_tmp,layer2_tmp)
	#(n x ma x mb x k) * (n x mc x k) -> (n x ma x mb x mc)
	layer = tf.einsum('nabk,nck->nabc',layer12_tmp,layer3_tmp)
	#layer=tf.reduce_sum(layer123,1)
	return layer

#===============================================================
def trilinear_discriminator(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	weights = tf.get_variable('trilinear_sibling_Weights', shape=[input1_size, input3_size, input2_size], initializer=tf.truncated_normal_initializer())
	weights2 = tf.get_variable('trilinear_grandparent_Weights', shape=[input3_size, input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	with tf.device('/device:GPU:2'):  
		weights3 = tf.get_variable('trilinear_coparent_Weights', shape=[input2_size, input3_size, input1_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	#pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (n x m x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (n x m x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	#=============================================================
	# First part, sibling potential
	# (ha x hc x hb)
	# (nm x d) * (d x d x d) -> (nm x d x d)
	layer_sib = tf.tensordot(layer1, weights, axes=[[1], [0]]) 
	# (nm x d x d) -> (nm x d x d)
	layer_sib = nn.reshape(layer_sib, [-1, bucket_size, input3_size, input2_size])
	# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer_sib = tf.einsum('abij,acj->abic', layer_sib, layer2)
	# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
	layer_sib = tf.einsum('abic,adi->abdc', layer_sib, layer3)
	# (n x mo x m x m) -> (n x m x m x m)
	layer_sib = nn.reshape(layer_sib, layer_shape + [bucket_size]*2)

	#=============================================================
	# Second part, grandparent potential
	# (hc x ha x hb)

	# (n x m x d) -> (nm x d)
	layer3 = nn.reshape(layer3, [-1, input3_size])
	# (n x m x d) -> (nm x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (nm x d)
	layer1 = tf.reshape(layer1, [-1, bucket_size, input1_size])

	# (nm x d) * (d x d x d) -> (nm x d x d)
	layer_gp = tf.tensordot(layer3, weights2, axes=[[1], [0]]) 
	# (nm x d x d) -> (nm x d x d)
	layer_gp = nn.reshape(layer_gp, [-1, bucket_size, input1_size, input2_size])
	# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer_gp = tf.einsum('abij,acj->abic', layer_gp, layer2)
	# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
	layer_gp = tf.einsum('abic,adi->abdc', layer_gp, layer1)
	# (n x mo x m x m) -> (n x m x m x m)
	layer_gp = nn.reshape(layer_gp, layer_shape + [bucket_size]*2)

	#=============================================================
	# Third part, coparent potential
	# (hb x hc x ha)
	with tf.device('/device:GPU:2'):  
		# (n x m x d) -> (nm x d)
		layer2 = nn.reshape(layer2, [-1, input2_size])
		# (n x m x d) -> (nm x d) why do this?
		layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
		# (n x m x d) -> (nm x d)
		layer1 = tf.reshape(layer1, [-1, bucket_size, input1_size])

		# (nm x d) * (d x d x d) -> (nm x d x d)
		layer_cop = tf.tensordot(layer2, weights3, axes=[[1], [0]]) 
		# (nm x d x d) -> (nm x d x d)
		layer_cop = nn.reshape(layer_cop, [-1, bucket_size, input3_size, input1_size])
		# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
		# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
		layer_cop = tf.einsum('abij,acj->abic', layer_cop, layer1)
		# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
		layer_cop = tf.einsum('abic,adi->abdc', layer_cop, layer3)
		# (n x mo x m x m) -> (n x m x m x m)
		layer_cop = nn.reshape(layer_cop, layer_shape + [bucket_size]*2)
	# wait! here is a problem, the tensor is different on each dimension! must adjust the dimension? or return three tensors to the outside?
	# TODO:
	# 1 dimension of each matrix
	# 2 parallelization in this part!
	# 3 for 1, here is a huge wrong(need transformation), then the previous simulation may have some problem!
	# layer_sib = (n x ma x mc x mb) -> (n x ma x mb x mc)
	layer_sib = tf.transpose(layer_sib, perm=[0,1,3,2])
	# layer_gp = (n x mc x ma x mb) -> (n x ma x mb x mc)
	layer_gp = tf.transpose(layer_gp, perm=[0,2,3,1])
	with tf.device('/device:GPU:2'):  
		# layer_cop = (n x mb x mc x ma) -> (n x ma x mb x mc)
		layer_cop = tf.transpose(layer_cop, perm=[0,3,1,2])
	# (x) here cannot add them up! there a something different here!
	#layer=layer_cop+layer_sib+layer_gp

	#return layer, weights
	#return layer
	return layer_sib, layer_gp, layer_cop

#===============================================================
def trilinear_discriminator2(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True):
	""""""
	# TODO: felt strange here...
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#pdb.set_trace()
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input3_size, input2_size], initializer=tf.truncated_normal_initializer())
	#(d x d) for layer1 and layer2 (d x d)->(m x m)
	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	#(d x d) for layer1 and layer3 (d x d)->(m x m)
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input1_size, input3_size], initializer=tf.truncated_normal_initializer())
	#(m x m x m) for  (m x m) * (m x m x m) * (m x m) ->(m x m x m)
	weights3 = tf.get_variable('trilinear_Weights3', shape=[bucket_size, bucket_size, bucket_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (nm x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (nm x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	
	# (nm x d) * (d x d) -> (nm x d)
	layer_temp1 = tf.matmul(layer1, weights1)
	layer_temp2 = tf.matmul(layer1, weights2)
	# (nm x d) -> (n x m x d)
	layer_temp1 = nn.reshape(layer_temp1, [-1, bucket_size, input2_size])
	layer_temp2 = nn.reshape(layer_temp2, [-1, bucket_size, input3_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	layer_temp1 = tf.matmul(layer_temp1, layer2, transpose_b=True)
	layer_temp2 = tf.matmul(layer_temp2, layer3, transpose_b=True)

	# (n x m x m) * (m x m x m) -> (n x m x m x m)
	# (nm x m) * (m x m^2) -> (nm x m^2)
	#layer_temp1 = nn.reshape(layer_temp1, [-1, bucket_size])
	#weights3_temp = nn.reshape(weights3, [-1, bucket_size*bucket_size])
	layer = tf.tensordot(layer_temp1, weights3, axes=[[-1], [0]])
	# (n x m x m x [m]) * (n x [m] x m) -> (n x m x m x m)
	layer = tf.einsum('nabc,ncd->nabd', layer, layer2)
	#layer = tf.tensordot(layer, layer_temp2, axes=[[-1],[]])
	layer = nn.reshape(layer, layer_shape + [bucket_size]*2)
	return layer

#===============================================================
def diagonal_bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	biases = tf.get_variable('Biases', shape=[1], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
	
	if add_linear:
		#(d) -> (d x 1)
		weights1 = tf.expand_dims(weights1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x 1) -> (nm x 1)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x 1) -> (n x m)
		lin_layer1 = nn.reshape(lin_layer1, layer_shape)
		# (n x m) -> (n x m x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		#(d) -> (d x 1)
		weights2 = tf.expand_dims(weights2, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x 1) -> (nm x 1)
		lin_layer2 = tf.matmul(lin_layer2, weights1)
		# (nm x 1) -> (n x m)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape)
		# (n x m) -> (n x 1 x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-2)
	
	# (n x m x d) -> (n x m x d)
	layer1 = nn.reshape(layer1, [-1, bucket_size, input_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	
	# (n x m x d) (*) (d) -> (n x m x d)
	layer = layer1 * weights
	# (n x m x d) * (n x m x d) -> (n x m x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x m x m) -> (n x m x m)
	layer = nn.reshape(layer, layer_shape + [bucket_size])
	if add_linear:
		# (n x m x m) + (n x 1 x m) + (n x m x 1) -> (n x m x m)
		layer += lin_layer1 + lin_layer2
	# (n x m x m) + () -> (n x m x m)
	layer += biases
	return layer

#===============================================================
def bilinear_attention(layer1, layer2, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+add_linear
	input2_size = layer2.get_shape().as_list()[-1]
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	original_layer1 = layer1
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
		noise_shape2 = tf.stack(layer_shape[:-2] + [1, input2_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	if add_linear:
		ones = tf.ones(ones_shape)
		layer1 = tf.concat([layer1, ones], -1)
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input2_size])
	
	# (nm x d) * (d x d) -> (nm x d)
	attn = tf.matmul(layer1, weights)
	# (nm x d) -> (n x m x d)
	attn = nn.reshape(attn, [-1, bucket_size, input2_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	attn = tf.matmul(attn, layer2, transpose_b=True)
	# (n x m x m) -> (n x m x m)
	attn = nn.reshape(attn, layer_shape + [bucket_size])
	# (n x m x m) -> (n x m x m)
	soft_attn = tf.nn.softmax(attn)
	# (n x m x m) * (n x m x d) -> (n x m x d)
	weighted_layer1 = tf.matmul(soft_attn, original_layer1)
	
	return attn, weighted_layer1
	
#===============================================================
def diagonal_bilinear_attention(layer1, layer2, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights2 = tf.get_variable('Weights2', shape=[input_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	original_layer1 = layer1
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
	
	if add_linear:
		#(d) -> (d x 1)
		weights2 = tf.expand_dims(weights2, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_attn2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x 1) -> (nm x 1)
		lin_attn2 = tf.matmul(lin_attn2, weights2)
		# (nm x 1) -> (n x m)
		lin_attn2 = nn.reshape(lin_attn2, layer_shape)
		# (n x m) -> (n x 1 x m)
		lin_attn2 = tf.expand_dims(lin_attn2, axis=-2)
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	
	# (nm x d) * (d) -> (nm x d)
	attn = layer1 * weights
	# (nm x d) -> (n x m x d)
	attn = nn.reshape(attn, [-1, bucket_size, input_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	attn = tf.matmul(attn, layer2, transpose_b=True)
	# (n x m x m) -> (n x m x m)
	attn = nn.reshape(attn, layer_shape + [bucket_size])
	if add_linear:
		# (n x m x m) + (n x 1 x m) -> (n x m x m)
		attn += lin_attn2
	# (n x m x m) -> (n x m x m)
	soft_attn = tf.nn.softmax(attn)
	# (n x m x m) * (n x m x d) -> (n x m x d)
	weighted_layer1 = tf.matmul(soft_attn, original_layer1)

	return attn, weighted_layer1
