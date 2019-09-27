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
from parser.structs.vocabs.pointer_generator import PointerGenerator
from . import mrp_vocabs as mv

from parser.neural import nn, nonlin, embeddings, classifiers, recurrent

import sys
sys.path.append('./THUMT')
import thumt.layers as layers
from thumt.models.rnnsearch import _decoder as seq2seq_decoder
# from THUMT.thumt.models.rnnsearch import _decoder as seq2seq_decoder

import pdb

class RNNDecoderVocab(TokenVocab):
	"""docstring for RNNDecoderVocab"""
	#_save_str = 'tokens'

	#=============================================================
	def __init__(self, *args, **kwargs):
		""""""
		if 'placeholder_shape' not in kwargs:
			kwargs['placeholder_shape'] = [None, None]
		super(RNNDecoderVocab, self).__init__(*args, **kwargs)
		return
	def forward(self, layers, decoder_embeddings, sentence_feat, token_weights, sequence_length, input_feed=None, target_copy_hidden_states=None, coverage=None,\
				 variable_scope=None, reuse=False, debug=False):
		"""
		decoder embeddings [batch_size, decoder_seq_length, embedding_size]
		layers: outputs of BiLSTM [batch_size, seq_length, hidden_size]
		sentence_feat: the final output state of RNN [num_encoder_layers, batch_size, hidden_size]
		token_weights: mask
		input_feed: None or [batch_size, 1, hidden_size]
		target_copy_hidden_states: None or [batch_size, seq_length, hidden_size]
		coverage: None or [batch_size, 1, encode_seq_length]
		"""

		#pdb.set_trace()
		with tf.variable_scope('Seq2SeqDecoder'):
			with tf.variable_scope('linear'):
				sentence_feat = classifiers.hidden(sentence_feat, self.recur_size,hidden_func=self.hidden_func,hidden_keep_prob=self.hidden_keep_prob)
			with tf.variable_scope('memory_linear'):
				layers = classifiers.hidden(layers, self.recur_size,hidden_func=self.hidden_func,hidden_keep_prob=self.hidden_keep_prob)
			with tf.variable_scope('embedding_linear'):
				decoder_embeddings = classifiers.hidden(decoder_embeddings, self.recur_size,hidden_func=self.hidden_func,hidden_keep_prob=self.hidden_keep_prob)
			result = seq2seq_decoder(self.cell,decoder_embeddings,layers,sequence_length,sentence_feat)
		return result

	def count(self, mrp):
		""""""
		# pdb.set_trace()
		mrp_file=json.load(open(mrp))
		for sentence_id in mrp_file:
			for current_data in mrp_file[sentence_id]['nodes']:
				token = current_data[self.field]
				self._count(token)
		self.index_by_counts()
		return True
	def count_mrp(self, mrp):
		""""""
		return True
	def _count(self, token):
		if not self.cased:
			token = token.lower()
		self.counts[token] += 1
		return
	def get_root(self):
		""""""
		return 0
	def add_sequence(self,tokens):
		indices=[x if x!='' else 0 for x in tokens]
		return indices
	@property
	def recur_size(self):
		return self._config.getint(self, 'recur_size')
	@property
	def get_nodes_path(self):
		return self._config.get('BaseNetwork', 'nodes_path')


class Seq2SeqIDVocab(RNNDecoderVocab, mv.NodeIDVocab):
	def set_placeholders(self, indices, feed_dict={}):
		""""""
		feed_dict[self.placeholder] = indices
		return feed_dict
	#=============================================================
	def get_bos(self):
		""""""
		
		return 0
	
	#=============================================================
	def get_eos(self):
		""""""
		
		return 0


class Seq2SeqNodeLabelPredictionVocab(TokenVocab, mv.LabelVocab):
	def __init__(self, *args, **kwargs):
		""""""
		kwargs['placeholder_shape'] = [None, None]
		super(Seq2SeqNodeLabelPredictionVocab, self).__init__(*args, **kwargs)
		
		return
	#=============================================================
	def get_bos(self):
		""""""
		
		return '<BOS>'
	
	#=============================================================
	def get_eos(self):
		""""""
		
		return '<EOS>'
	def forward(self, hiddens, source_attentions, target_attentions, pointer_generator_inputs, invalid_indexes=None,\
				 variable_scope=None, reuse=False, debug=False):
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

		#pdb.set_trace()
		# target=self.placeholder['vocab_targets']
		# copy_targets=self.placeholder['copy_targets']
		# coref_targets=self.placeholder['coref_targets']
		with tf.variable_scope('Seq2SeqNodeLabelPredictionVocab'):
			source_attention_maps=pointer_generator_inputs['SrcCopyMap']
			target_attention_maps=pointer_generator_inputs['TgtCopyMap'][:,1:]
			outputs=self.predictor.forward(hiddens, source_attentions, source_attention_maps, target_attentions, target_attention_maps, invalid_indexes=None,debug=debug)
			copy_targets=pointer_generator_inputs['SrcCopyIndices'][:,1:]
			coref_targets=pointer_generator_inputs['TgtCopyIndices'][:,1:]
			# pdb.set_trace()
			loss_outputs = self.predictor.compute_loss(outputs['probabilities'],outputs['predictions'],self.placeholder,copy_targets,outputs['source_dynamic_vocab_size'],coref_targets,outputs['source_dynamic_vocab_size'],None,target_attentions,debug=debug)
			outputs.update(loss_outputs)
			outputs['loss'] = outputs['loss']*self.loss_interpolation
			# outputs['loss']=tf.zeros(1,tf.float32)[0]
			# outputs['n_correct_tokens']=tf.zeros(1,tf.float32)[0]
			# outputs['n_correct_sequences'] = tf.zeros(1,tf.float32)[0]
		return outputs
	def decode(self, memory_bank, mask, states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes, decoder_inputs):
		# [batch_size, 1]
		batch_size = tf.shape(memory_bank)[0]

		tokens = tt.ones([batch_size, 1]) * self.index('<BOS>')
		pos_tags = torch.ones(batch_size, 1) * self.index('<EOS>')
		corefs = torch.zeros(batch_size, 1)

		decoder_input_history = []
		decoder_outputs = []
		rnn_outputs = []
		copy_attentions = []
		coref_attentions = []
		predictions = []
		coref_indexes = []
		decoder_mask = []

		input_feed = None
		coref_inputs = []

		# A sparse indicator matrix mapping each node to its index in the dynamic vocab.
		# Here the maximum size of the dynamic vocab is just max_decode_length.
		coref_attention_maps = tf.cast(tf.zeros([batch_size, self.max_decode_length, self.max_decode_length + 1]), tf.float32)
		# A matrix D where the element D_{ij} is for instance i the real vocab index of
		# the generated node at the decoding step `i'.
		coref_vocab_maps = tf.zeros([batch_size, self.max_decode_length + 1])

		coverage = None
		if self.use_coverage:
			coverage = memory_bank.new_zeros(batch_size, 1, memory_bank.size(1))

		for step_i in range(self.max_decode_length):

			# 2. Decode one step.
			decoder_output_dict = self.decoder(
				decoder_inputs, memory_bank, mask, states, input_feed, coref_inputs, coverage)
			_decoder_outputs = decoder_output_dict['decoder_hidden_states']
			_rnn_outputs = decoder_output_dict['rnn_hidden_states']
			_copy_attentions = decoder_output_dict['source_copy_attentions']
			_coref_attentions = decoder_output_dict['target_copy_attentions']
			states = decoder_output_dict['last_hidden_state']
			input_feed = decoder_output_dict['input_feed']
			coverage = decoder_output_dict['coverage']

			# 3. Run pointer/generator.
			if step_i == 0:
				_coref_attention_maps = coref_attention_maps[:, :step_i + 1]
			else:
				_coref_attention_maps = coref_attention_maps[:, :step_i]

			generator_output = self.generator(
				_decoder_outputs, _copy_attentions, copy_attention_maps,
				_coref_attentions, _coref_attention_maps, invalid_indexes)
			_predictions = generator_output['predictions']

			# 4. Update maps and get the next token input.
			tokens, _predictions, pos_tags, corefs, _mask = self._update_maps_and_get_next_input(
				step_i,
				generator_output['predictions'].squeeze(1),
				generator_output['source_dynamic_vocab_size'],
				coref_attention_maps,
				coref_vocab_maps,
				copy_vocabs,
				decoder_mask,
				tag_luts,
				invalid_indexes
			)

			# 5. Update variables.
			decoder_input_history += [decoder_inputs]
			decoder_outputs += [_decoder_outputs]
			rnn_outputs += [_rnn_outputs]

			copy_attentions += [_copy_attentions]
			coref_attentions += [_coref_attentions]

			predictions += [_predictions]
			# Add the coref info for the next input.
			coref_indexes += [corefs]
			# Add the mask for the next input.
			decoder_mask += [_mask]

		# 6. Do the following chunking for the graph decoding input.
		# Exclude the hidden state for BOS.
		decoder_input_history = torch.cat(decoder_input_history[1:], dim=1)
		decoder_outputs = torch.cat(decoder_outputs[1:], dim=1)
		rnn_outputs = torch.cat(rnn_outputs[1:], dim=1)
		# Exclude coref/mask for EOS.
		# TODO: Answer "What if the last one is not EOS?"
		predictions = torch.cat(predictions[:-1], dim=1)
		coref_indexes = torch.cat(coref_indexes[:-1], dim=1)
		decoder_mask = 1 - torch.cat(decoder_mask[:-1], dim=1)

		return dict(
			# [batch_size, max_decode_length]
			predictions=predictions,
			coref_indexes=coref_indexes,
			decoder_mask=decoder_mask,
			# [batch_size, max_decode_length, hidden_size]
			decoder_inputs=decoder_input_history,
			decoder_memory_bank=decoder_outputs,
			decoder_rnn_memory_bank=rnn_outputs,
			# [batch_size, max_decode_length, encoder_length]
			copy_attentions=copy_attentions,
			coref_attentions=coref_attentions
		)


class Seq2SeqSrcCopyMapVocab(RNNDecoderVocab, mv.SrcCopyMapVocab):
	def __init__(self, *args, **kwargs):
		""""""
		self._depth=-2
		kwargs['placeholder_shape'] = [None, None, None]
		super(Seq2SeqSrcCopyMapVocab, self).__init__(*args, **kwargs)
		return

class Seq2SeqTgtCopyMapVocab(RNNDecoderVocab, mv.TgtCopyMapVocab):
	def __init__(self, *args, **kwargs):
		""""""
		self._depth=-2
		kwargs['placeholder_shape'] = [None, None, None]
		super(Seq2SeqTgtCopyMapVocab, self).__init__(*args, **kwargs)
		return

class Seq2SeqSrcCopyIndicesVocab(RNNDecoderVocab, mv.SrcCopyIndicesVocab):
	def __init__(self, *args, **kwargs):
		""""""
		kwargs['placeholder_shape'] = [None, None]
		super(Seq2SeqSrcCopyIndicesVocab, self).__init__(*args, **kwargs)
		return

class Seq2SeqTgtCopyIndicesVocab(RNNDecoderVocab, mv.TgtCopyIndicesVocab):
	def __init__(self, *args, **kwargs):
		""""""
		kwargs['placeholder_shape'] = [None, None]
		super(Seq2SeqTgtCopyIndicesVocab, self).__init__(*args, **kwargs)
		return

class Seq2SeqDecoderVocab(RNNDecoderVocab, mv.WordVocab):
	def __init__(self, *args, **kwargs):
		""""""
		kwargs['placeholder_shape'] = [None, None]
		super(Seq2SeqDecoderVocab, self).__init__(*args, **kwargs)
		self.cell = layers.rnn_cell.LegacyGRUCell(self.recur_size)
		# self.predictor = PointerGenerator(self, input_size, switch_input_size, vocab_size, vocab_pad_idx, force_copy)
		return
	#=============================================================
	def get_bos(self):
		""""""
		
		return 0
	
	#=============================================================
	def get_eos(self):
		""""""
		
		return 0
	def forward(self, layers, decoder_embeddings, sentence_feat, token_weights, sequence_length, input_feed=None, target_copy_hidden_states=None, coverage=None,\
				 variable_scope=None, reuse=False, debug=False):
		"""
		decoder embeddings [batch_size, decoder_seq_length, embedding_size]
		layers: outputs of BiLSTM [batch_size, seq_length, hidden_size]
		sentence_feat: the final output state of RNN [num_encoder_layers, batch_size, hidden_size]
		token_weights: mask
		input_feed: None or [batch_size, 1, hidden_size]
		target_copy_hidden_states: None or [batch_size, seq_length, hidden_size]
		coverage: None or [batch_size, 1, encode_seq_length]
		"""

		with tf.variable_scope('Seq2SeqDecoder'):
			with tf.variable_scope('linear'):
				sentence_feat = classifiers.hidden(sentence_feat, self.recur_size,hidden_func=self.hidden_func,hidden_keep_prob=self.hidden_keep_prob)
			with tf.variable_scope('memory_linear'):
				layers = classifiers.hidden(layers, self.recur_size,hidden_func=self.hidden_func,hidden_keep_prob=self.hidden_keep_prob)
			with tf.variable_scope('embedding_linear'):
				decoder_embeddings = classifiers.hidden(decoder_embeddings, self.recur_size,hidden_func=self.hidden_func,hidden_keep_prob=self.hidden_keep_prob)
			result = seq2seq_decoder(self.cell,decoder_embeddings,layers,sequence_length,sentence_feat)
		return result
		
class Seq2SeqAnchorPredictionVocab(RNNDecoderVocab, mv.AnchorVocab):
	pass
class Seq2SeqGraphTokenVocab(GraphTokenVocab, mv.SemrelVocab):
	def count(self, mrp):
		""""""
		# pdb.set_trace()
		mrp_file=json.load(open(mrp))
		for sentence_id in mrp_file:
			for current_data in mrp_file[sentence_id]['nodes']:
				token = current_data[self.field]
				self._count(token)
		self.index_by_counts()
		return True
	def _count(self, node):
		if node not in ('_', ''):
			node = node.split('|')
			for edge in node:
				edge = edge.split(':', 1)
				head, rel = edge
				self.counts[rel] += 1
		return
	#=============================================================
	def get_bos(self):
		""""""
		
		return '_'
	
	#=============================================================
	def get_eos(self):
		""""""
		
		return '_'
	#=============================================================
	# def add(self, token):
	# 	""""""
	# 	indices=self.index(token)
	# 	indices=[(index[0]+1,index[1]) for index in indices]
	# 	return indices
class Seq2SeqGraphIndexVocab(GraphIndexVocab, mv.SemheadVocab):
	def count(self, mrp):
		""""""
		# pdb.set_trace()
		mrp_file=json.load(open(mrp))
		for sentence_id in mrp_file:
			for current_data in mrp_file[sentence_id]['nodes']:
				token = current_data[self.field]
				self._count(token)
		self.index_by_counts()
		return True
	def _count(self, node):
		if node not in ('_', ''):
			node = node.split('|')
			for edge in node:
				edge = edge.split(':', 1)
				head, rel = edge
				self.counts[rel] += 1
		return
	# def add(self, token):
	# 	""""""
	# 	indices=self.index(token)
	# 	indices=[index+1 for index in indices]
	# 	return indices
	#=============================================================
	def get_bos(self):
		""""""
		
		return '_'
	
	#=============================================================
	def get_eos(self):
		""""""
		
		return '_'
class Seq2SeqSecondOrderGraphIndexVocab(GraphSecondIndexVocab, mv.SemheadVocab):
	def count(self, mrp):
		""""""
		# pdb.set_trace()
		mrp_file=json.load(open(mrp))
		for sentence_id in mrp_file:
			for current_data in mrp_file[sentence_id]['nodes']:
				token = current_data[self.field]
				self._count(token)
		self.index_by_counts()
		return True
	def _count(self, node):
		if node not in ('_', ''):
			node = node.split('|')
			for edge in node:
				edge = edge.split(':', 1)
				head, rel = edge
				self.counts[rel] += 1
		return
	#=============================================================
	def get_bos(self):
		""""""
		
		return '_'
	
	#=============================================================
	def get_eos(self):
		""""""
		
		return '_'
	# def add(self, token):
	# 	""""""
	# 	indices=self.index(token)
	# 	indices=[index+1 for index in indices]
	# 	return indices