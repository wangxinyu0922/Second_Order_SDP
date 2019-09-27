from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import json
import os
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.structs.vocabs.base_vocabs import CountVocab
from parser.structs.vocabs.token_vocabs import TokenVocab,GraphTokenVocab
from parser.structs.vocabs.index_vocabs import IndexVocab,GraphIndexVocab
from parser.structs.vocabs.second_order_vocab import GraphSecondIndexVocab
from . import mrp_vocabs as mv

from parser.neural import nn, nonlin, embeddings, classifiers, recurrent

import sys
sys.path.append('./THUMT')
import thumt.layers as layers
from thumt.models.rnnsearch import _decoder as seq2seq_decoder
# from THUMT.thumt.models.rnnsearch import _decoder as seq2seq_decoder

import pdb


class PointerGenerator(object):
	"""docstring for Pointer_Generator"""
	def __init__(self, input_size, switch_input_size, vocab_size, vocab_pad_idx, force_copy, hidden_func, hidden_keep_prob):
		super(PointerGenerator, self).__init__()
		self.hidden_keep_prob=hidden_keep_prob
		self.hidden_func = hidden_func
		self.vocab_size = vocab_size
		self.vocab_pad_idx = vocab_pad_idx

		self.force_copy = force_copy

		self.eps = 1e-20
	def forward(self, hiddens, source_attentions, source_attention_maps, target_attentions, target_attention_maps, invalid_indexes=None, debug=False):
		"""
		Compute a distribution over the target dictionary
		extended by the dynamic dictionary implied by copying target nodes.

		:param hiddens: decoder outputs, [batch_size, num_target_nodes, hidden_size]
		:param source_attentions: attention of each source node,
			[batch_size, num_target_nodes, num_source_nodes]
		:param source_attention_maps: a sparse indicator matrix
			mapping each source node to its index in the dynamic vocabulary.
			[batch_size, num_source_nodes, dynamic_vocab_size]
		:param target_attentions: attention of each target node,
			[batch_size, num_target_nodes, num_target_nodes]
		:param target_attention_maps: a sparse indicator matrix
			mapping each target node to its index in the dynamic vocabulary.
			[batch_size, num_target_nodes, dynamic_vocab_size]
		:param invalid_indexes: indexes which are not considered in prediction.
		"""
		# pdb.set_trace()
		with tf.variable_scope('Pointer_Generator'):
			batch_size, num_target_nodes, hidden_size = nn.get_sizes(hiddens)
			source_dynamic_vocab_size = tf.shape(source_attention_maps)[2]
			target_dynamic_vocab_size = tf.shape(target_attention_maps)[2]
			hiddens = nn.reshape(hiddens, [batch_size * num_target_nodes, hidden_size])
			with tf.variable_scope('linear_pointer'):
				temp_layer = classifiers.hidden(hiddens, 3,hidden_func=self.hidden_func,hidden_keep_prob=self.hidden_keep_prob)

			# Pointer probability.
			p = tf.nn.softmax(temp_layer, axis=1)
			p_temp = nn.reshape(p, [batch_size, num_target_nodes, 3])
			p_copy_source = p_temp[:, :, 0]
			p_copy_target = p_temp[:, :, 1]
			p_generate = p_temp[:, :, 2]

			# Probability distribution over the vocabulary.
			# [batch_size * num_target_nodes, vocab_size]
			with tf.variable_scope('linear'):
				scores = classifiers.hidden(hiddens, self.vocab_size, hidden_func=self.hidden_func,hidden_keep_prob=self.hidden_keep_prob)

			#scores[:, self.vocab_pad_idx] = -float('inf')
			# [batch_size * num_target_nodes, 0]=-inf
			inf_mask=tf.zeros_like(scores)-99999999.0
			minus_mask=1-tf.cast(nn.greater(tf.range(self.vocab_size), 0),dtype=tf.float32)[None,:]
			mask_val=(minus_mask*inf_mask)
			scores+=mask_val
			# inf_tensor = tf.constant(-float('inf'), shape=tf.shape(pad_idx))
			# tf.assign(scores[:,0],-float('inf'))
			# [batch_size, num_target_nodes, vocab_size]
			scores = nn.reshape(scores, [batch_size, num_target_nodes, -1])
			vocab_probs = tf.nn.softmax(scores, axis=-1)

			scaled_vocab_probs = vocab_probs * p_generate[:,:,None]


			# [batch_size, num_target_nodes, num_source_nodes]
			scaled_source_attentions = source_attentions * p_copy_source[:,:,None]
			# [batch_size, num_target_nodes, dynamic_vocab_size]
			scaled_copy_source_probs = tf.matmul(scaled_source_attentions, tf.cast(source_attention_maps,dtype=tf.float32))

			# Probability distribution over the dynamic vocabulary.
			# [batch_size, num_target_nodes, num_target_nodes]
			# TODO: make sure for target_node_i, its attention to target_node_j >= target_node_i
			# should be zero.
			scaled_target_attentions = target_attentions * p_copy_target[:,:,None]
			# [batch_size, num_target_nodes, dymanic_vocab_size]
			scaled_copy_target_probs = tf.matmul(scaled_target_attentions, tf.cast(target_attention_maps,dtype=tf.float32))

			if invalid_indexes:
				if invalid_indexes.get('vocab', None) is not None:
					vocab_invalid_indexes = invalid_indexes['vocab']
					for i, indexes in enumerate(vocab_invalid_indexes):
						for index in indexes:
							zero_tensor = tf.constant(0.0, shape=tf.shape(scaled_vocab_probs[i, :, index]))
							tf.assign(scaled_vocab_probs[i, :, index],zero_tensor)

				if invalid_indexes.get('source_copy', None) is not None:
					source_copy_invalid_indexes = invalid_indexes['source_copy']
					for i, indexes in enumerate(source_copy_invalid_indexes):
						for index in indexes:
							zero_tensor = tf.constant(0.0, shape=tf.shape(scaled_copy_source_probs[i, :, index]))
							tf.assign(scaled_copy_source_probs[i, :, index],zero_tensor)

			# [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
			probs = tf.concat([scaled_vocab_probs,scaled_copy_source_probs,scaled_copy_target_probs], 2)
			
			# Set the probability of coref NA to 0.
			_probs = tf.identity(probs)
			#_probs[:, :, self.vocab_size + source_dynamic_vocab_size] = 0
			probs_mask=tf.cast(nn.equal(tf.range(self.vocab_size+source_dynamic_vocab_size+target_dynamic_vocab_size), self.vocab_size+source_dynamic_vocab_size),dtype=tf.float32)[None,None,:]
			_probs-=(probs_mask*_probs)

			predictions = tf.cast(tf.argmax(_probs,2),tf.int32)
			# _, predictions = _probs.max(2)
			output=dict(
				probabilities=probs,
				predictions=predictions,
				source_dynamic_vocab_size=source_dynamic_vocab_size,
				target_dynamic_vocab_size=target_dynamic_vocab_size
			)
			if debug:
				output['probs']=probs
				output['probs2']=_probs
				output['scores']=scores
				output['inf_mask']=inf_mask
				output['minus_mask']=minus_mask
				output['probs_mask']=probs_mask
				output['temp_layer']=temp_layer
				output['p_copy_source']=p_copy_source
				output['source_attentions']=source_attentions
				output['scaled_source_attentions']=scaled_source_attentions
				output['scaled_copy_target_probs']=scaled_copy_target_probs
				output['p']=p
			return output
	def compute_loss(self, probs, predictions, generate_targets,
					 source_copy_targets, source_dynamic_vocab_size,
					 target_copy_targets, target_dynamic_vocab_size,
					 coverage_records, copy_attentions, debug=False):
		"""
		Priority: target_copy > source_copy > generate

		:param probs: probability distribution,
			[batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
		:param predictions: [batch_size, num_target_nodes]
		:param generate_targets: target node index in the vocabulary,
			[batch_size, num_target_nodes]
		:param source_copy_targets:  target node index in the dynamic vocabulary,
			[batch_size, num_target_nodes]
		:param source_dynamic_vocab_size: int
		:param target_copy_targets:  target node index in the dynamic vocabulary,
			[batch_size, num_target_nodes]
		:param target_dynamic_vocab_size: int
		:param coverage_records: None or a tensor recording source-side coverages.
			[batch_size, num_target_nodes, num_source_nodes]
		:param copy_attentions: [batch_size, num_target_nodes, num_source_nodes]
		"""
		# pdb.set_trace()
		generate_targets=generate_targets[:,1:]
		non_pad_mask = tf.cast(tf.not_equal(generate_targets,self.vocab_pad_idx),tf.int32)

		#source_copy_mask = source_copy_targets.ne(1) & source_copy_targets.ne(0)  # 1 is the index for unknown words
		source_copy_mask = tf.cast(tf.logical_and(tf.not_equal(source_copy_targets,1),tf.not_equal(source_copy_targets,0)),tf.int32)
		non_source_copy_mask = 1 - source_copy_mask

		target_copy_mask = tf.cast(tf.not_equal(target_copy_targets,0),tf.int32)
		non_target_copy_mask = 1 - target_copy_mask

		# [batch_size, num_target_nodes, 1]
		target_copy_targets_with_offset = tf.expand_dims(target_copy_targets,2) + self.vocab_size + source_dynamic_vocab_size
		#=================================================
		# [batch_size, num_target_nodes]
		target_copy_target_probs = tf.squeeze(tf.batch_gather(probs,target_copy_targets_with_offset),2)
		target_copy_target_probs = target_copy_target_probs*tf.cast(target_copy_mask,dtype=tf.float32)
		# target_copy_target_probs = probs.gather(dim=2, index=target_copy_targets_with_offset).squeeze(2)
		# target_copy_target_probs = target_copy_target_probs.mul(target_copy_mask.float())

		# [batch_size, num_target_nodes, 1]
		source_copy_targets_with_offset = tf.expand_dims(source_copy_targets,2) + self.vocab_size
		# [batch_size, num_target_nodes]
		source_copy_target_probs = tf.squeeze(tf.batch_gather(probs,source_copy_targets_with_offset),2)
		source_copy_target_probs = source_copy_target_probs*tf.cast(non_target_copy_mask,dtype=tf.float32)*tf.cast(source_copy_mask,dtype=tf.float32)

		# [batch_size, num_target_nodes]
		generate_target_probs = tf.squeeze(tf.batch_gather(probs,generate_targets[:,:,None]),2)

		# Except copy-oov nodes, all other nodes should be copied.
		likelihood = target_copy_target_probs + source_copy_target_probs + \
					 generate_target_probs * tf.cast(non_target_copy_mask,dtype=tf.float32) * tf.cast(non_source_copy_mask,dtype=tf.float32)
		num_tokens = tf.reduce_sum(tf.cast(non_pad_mask,tf.int32))

		if not self.force_copy:
			non_generate_oov_mask = tf.not_equal(generate_targets,1)
			additional_generate_mask = (non_target_copy_mask & source_copy_mask & non_generate_oov_mask)
			likelihood = likelihood + generate_target_probs * tf.cast(additional_generate_mask,dtype=tf.float32)
			num_tokens += tf.reduce_sum(tf.cast(additional_generate_mask,tf.float32))

		# Add eps for numerical stability.
		likelihood = likelihood + self.eps

		coverage_loss = 0
		if coverage_records is not None:
			coverage_loss = tf.reduce_sum(tf.reduce_min(coverage_records, copy_attentions), 2) * tf.cast(non_pad_mask, dtype=tf.float32)

		# Drop pads.
		loss = -tf.log(likelihood) * tf.cast(non_pad_mask, dtype=tf.float32) + coverage_loss

		# Mask out copy targets for which copy does not happen.
		targets = tf.squeeze(target_copy_targets_with_offset,2) * target_copy_mask + \
				  tf.squeeze(source_copy_targets_with_offset,2) * non_target_copy_mask * source_copy_mask + \
				  generate_targets * non_target_copy_mask * non_source_copy_mask
		targets = targets * non_pad_mask

		pred_eq = tf.cast(tf.equal(predictions,targets),tf.int32) * non_pad_mask

		num_non_pad = tf.reduce_sum(non_pad_mask)
		num_correct_pred = tf.reduce_sum(pred_eq)

		num_target_copy = tf.reduce_sum(target_copy_mask * non_pad_mask)
		num_correct_target_copy = tf.reduce_sum(pred_eq * target_copy_mask)
		num_correct_target_point = tf.reduce_sum(tf.cast(tf.greater_equal(predictions,self.vocab_size + source_dynamic_vocab_size),tf.int32)*target_copy_mask*non_pad_mask)

		num_source_copy = tf.reduce_sum(source_copy_mask * non_target_copy_mask * non_pad_mask)
		num_correct_source_copy = tf.reduce_sum(pred_eq * non_target_copy_mask * source_copy_mask)
		num_correct_source_point = tf.reduce_sum(tf.cast(tf.greater_equal(predictions,self.vocab_size), tf.int32) * tf.cast(tf.less(predictions,self.vocab_size + source_dynamic_vocab_size), tf.int32) * non_target_copy_mask * source_copy_mask * non_pad_mask)

		# self.metrics(loss.sum().item(), num_non_pad, num_correct_pred,
		#              num_source_copy, num_correct_source_copy, num_correct_source_point,
		#              num_target_copy, num_correct_target_copy, num_correct_target_point
		#              )
		# num_correct_sequences = pred_eq = tf.cast(tf.equal(predictions,targets),tf.int32)
		# correct_tokens = nn.equal(targets, predictions) * token_weights
		# pdb.set_trace()
		# (n x m) -> (n)
		tokens_per_sequence = tf.reduce_sum(non_pad_mask, axis=-1)
		# (n x m) -> (n)
		correct_tokens_per_sequence = tf.reduce_sum(pred_eq, axis=-1)
		# (n), (n) -> (n)
		correct_sequences = tf.cast(nn.equal(tokens_per_sequence, correct_tokens_per_sequence),tf.float32)
		outputs=dict(
			loss=tf.reduce_sum(loss)/tf.cast(num_tokens,dtype=tf.float32),
			total_loss=tf.reduce_sum(loss),
			n_tokens=num_tokens,
			predictions=predictions,
			n_correct_tokens=num_correct_pred,
			n_correct_sequences=tf.reduce_sum(correct_sequences)
		)
		if debug:
			outputs['pred_eq']=pred_eq
			outputs['targets']=targets
			outputs['target_copy_targets_with_offset']=target_copy_targets_with_offset
			outputs['target_copy_mask']=target_copy_mask
			outputs['source_copy_targets_with_offset']=source_copy_targets_with_offset
			outputs['non_target_copy_mask']=non_target_copy_mask
			outputs['source_copy_mask']=source_copy_mask
			outputs['generate_targets']=generate_targets
			outputs['num_correct_pred']=num_correct_pred
			outputs['correct_tokens_per_sequence']=correct_tokens_per_sequence
		return outputs

	def _update_maps_and_get_next_input(
			self, step, predictions, copy_vocab_size, coref_attention_maps, coref_vocab_maps,
			copy_vocabs, masks, tag_luts, invalid_indexes):
		"""Dynamically update/build the maps needed for copying.
		#suppose here all inputs are in numpy format
		:param step: the decoding step, int.
		:param predictions: [batch_size]
		:param copy_vocab_size: int.
		:param coref_attention_maps: [batch_size, max_decode_length, max_decode_length]
		:param coref_vocab_maps:  [batch_size, max_decode_length]
		:param copy_vocabs: a list of dynamic vocabs.
		:param masks: a list of [batch_size] tensors indicating whether EOS has been generated.
			if EOS has has been generated, then the mask is `1`.
		:param tag_luts: a dict mapping key to a list of dicts mapping a source token to a POS tag.
		:param invalid_indexes: a dict storing invalid indexes for copying and generation.
		:return:
		"""
		vocab_size = self.vocab_size
		batch_size = predictions.shape(0)

		batch_index = np.array(range(batch_size))
		step_index = np.ones(predictions.shape) * step

		gen_mask = predictions < vocab_size
		copy_mask = np.greater_equal(predictions,vocab_size) * np.less(predictions,vocab + copy_vocab_size)
		coref_mask = np.greater_equal(coref_mask, vocab_size + copy_vocab_size)

		# 1. Update coref_attention_maps
		# Get the coref index.
		coref_index = (predictions - vocab_size - copy_vocab_size)
		# Fill the place where copy didn't happen with the current step,
		# which means that the node doesn't refer to any precedent, it refers to itself.
		coref_index = coref_index * coref_mask + (1 - coref_mask) * (step+1)

		coref_attention_maps[batch_index, step_index, coref_index] = 1

		# 2. Compute the next input.
		# coref_predictions have the dynamic vocabulary index, and OOVs are set to zero.
		coref_predictions = (predictions - vocab_size - copy_vocab_size) * coref_mask.as_type(np.int32)
		# Get the actual coreferred token's index in the gen vocab.
		coref_predictions = np.take(coref_vocab_maps, coref_predictions[:,None], 1).squeeze(1)

		# If a token is copied from the source side, we look up its index in the gen vocab.
		copy_predictions = (predictions - vocab_size) * copy_mask.as_type(np.int32)
		# pos_tags = torch.full_like(predictions, self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'pos_tags'))
		for i, index in enumerate(copy_predictions.tolist()):
			copied_token = copy_vocabs[i].get_token_from_idx(index)
			if index != 0:
				pos_tags[i] = self.vocab.get_token_index(
					tag_luts[i]['pos'][copied_token], 'pos_tags')
				if False: # is_abstract_token(copied_token):
					invalid_indexes['source_copy'][i].add(index)
			copy_predictions[i] = self.vocab.get_token_index(copied_token, 'decoder_token_ids')

		for i, index in enumerate(
				(predictions * gen_mask.long() + coref_predictions * coref_mask.long()).tolist()):
			if index != 0:
				token = self.vocab.get_token_from_index(index, 'decoder_token_ids')
				src_token = find_similar_token(token, list(tag_luts[i]['pos'].keys()))
				if src_token is not None:
					pos_tags[i] = self.vocab.get_token_index(
						tag_luts[i]['pos'][src_token], 'pos_tag')
				if False: # is_abstract_token(token):
					invalid_indexes['vocab'][i].add(index)

		next_input = coref_predictions * coref_mask.long() + \
					 copy_predictions * copy_mask.long() + \
					 predictions * gen_mask.long()

		# 3. Update dynamic_vocab_maps
		# Here we update D_{step} to the index in the standard vocab.
		coref_vocab_maps[batch_index, step_index + 1] = next_input

		# 4. Get the coref-resolved predictions.
		coref_resolved_preds = coref_predictions * coref_mask.long() + predictions * (1 - coref_mask).long()

		# 5. Get the mask for the current generation.
		has_eos = torch.zeros_like(gen_mask)
		if len(masks) != 0:
			has_eos = torch.cat(masks, 1).long().sum(1).gt(0)
		mask = next_input.eq(self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')) | has_eos

		return (next_input.unsqueeze(1),
				coref_resolved_preds.unsqueeze(1),
				pos_tags.unsqueeze(1),
				coref_index.unsqueeze(1),
				mask.unsqueeze(1))