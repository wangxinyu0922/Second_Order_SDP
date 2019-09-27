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

import re
import time
import os
import pickle as pkl
import curses
import codecs

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from debug.timer import Timer

from parser.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser.graph_outputs import GraphOutputs, TrainOutputs, DevOutputs
from parser.structs import conllu_dataset
from parser.structs import vocabs
from parser.neural.optimizers import AdamOptimizer, AMSGradOptimizer

import uuid
uid = uuid.uuid4().hex[:6]
import pdb

import sys, os

from tensorflow.python import debug as tf_debug

# Disable
def blockPrint():
		sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
		sys.stdout = sys.__stdout__


#***************************************************************
class BaseNetwork(object):
	""""""

	_evals = set()

	#=============================================================
	def __init__(self, input_networks=set(), config=None):
		""""""
		#pdb.set_trace()
		self._config = config
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

		self._input_networks = input_networks
		input_network_classes = set(input_network.classname for input_network in self._input_networks)
		assert input_network_classes == set(self.input_network_classes), 'Not all input networks were passed in to {}'.format(self.classname)

		extant_vocabs = {}
		for input_network in self.input_networks:
			for vocab in input_network.vocabs:
				if vocab.classname in extant_vocabs:
					assert vocab is extant_vocabs[vocab.classname], "Two input networks have different instances of {}".format(vocab.classname)
				else:
					extant_vocabs[vocab.classname] = vocab

		if 'IDIndexVocab' in extant_vocabs:
			self._id_vocab = extant_vocabs['IDIndexVocab']
		else:
			self._id_vocab = vocabs.IDIndexVocab(config=config)
			extant_vocabs['IDIndexVocab'] = self._id_vocab

		self._input_vocabs = []
		self._decoder_vocabs = []
		for input_vocab_classname in self.input_vocab_classes:
			if input_vocab_classname in extant_vocabs:
				self._input_vocabs.append(extant_vocabs[input_vocab_classname])
			else:
				VocabClass = getattr(vocabs, input_vocab_classname)
				vocab = VocabClass(config=config)
				if hasattr(vocab,'conllu_idx'):
					# pdb.set_trace()
					vocab.load() or vocab.count(self.train_conllus)
					self._input_vocabs.append(vocab)
				else:
					#pdb.set_trace()
					vocab.load() or vocab.count_mrp(self.get_nodes_path)
					#vocab.count(self.get_nodes_path)
					self._decoder_vocabs.append(vocab)
				extant_vocabs[input_vocab_classname] = vocab
			if 'Bert' in input_vocab_classname:
				self.use_bert=True
				self.bertvocab=self._input_vocabs[-1]
				self.pretrained_bert=self.bertvocab.get_pretrained
			else:
				self.use_bert=False
		#pdb.set_trace()
		self._output_vocabs = []
		self.use_seq2seq=False
		for output_vocab_classname in self.output_vocab_classes:
			if 'seq2seq' in output_vocab_classname.lower():
				self.use_seq2seq=True
			if output_vocab_classname in extant_vocabs:
				self._output_vocabs.append(extant_vocabs[output_vocab_classname])
			else:#create index vocabs and token vocabs (network)
				VocabClass = getattr(vocabs, output_vocab_classname)
				vocab = VocabClass(config=config)
				if hasattr(vocab,'conllu_idx'):
					vocab.load() or vocab.count(self.train_conllus)
				else:
					vocab.load() or vocab.count(self.get_nodes_path)
				self._output_vocabs.append(vocab)
				extant_vocabs[output_vocab_classname] = vocab
		if self.use_seq2seq:
			# pdb.set_trace()
			self._node_id_vocab = vocabs.Seq2SeqIDVocab(config=config)
			# self._output_vocabs.append(self._node_id_vocab)
			extant_vocabs[self._node_id_vocab.classname] = self._node_id_vocab
		self._throughput_vocabs = []
		for throughput_vocab_classname in self.throughput_vocab_classes:
			if throughput_vocab_classname in extant_vocabs:
				self._throughput_vocabs.append(extant_vocabs[throughput_vocab_classname])
			else:
				VocabClass = getattr(vocabs, throughput_vocab_classname)
				vocab = VocabClass(config=config)
				if hasattr(vocab,'conllu_idx'):
					vocab.load() or vocab.count(self.train_conllus)
				else:
					vocab.load() or vocab.count(vocab.get_nodes_path)
				self._throughput_vocabs.append(vocab)
				extant_vocabs[throughput_vocab_classname] = vocab

		with tf.variable_scope(self.classname, reuse=False):
			self.global_step = tf.Variable(0., trainable=False, name='Global_step')
		self._vocabs = set(extant_vocabs.values())
		return

	#=============================================================
	def train(self, load=False, noscreen=False, debug=False, nornn=False):
		""""""
		tf.set_random_seed(12345)
		trainset = conllu_dataset.CoNLLUTrainset(self.vocabs,
																						 config=self._config)
		devset = conllu_dataset.CoNLLUDevset(self.vocabs,
																				 config=self._config)
		testset = conllu_dataset.CoNLLUTestset(self.vocabs,
																					 config=self._config)
		#pdb.set_trace()
		factored_deptree = None
		factored_semgraph = None
		for vocab in self.output_vocabs:
			if vocab.field == 'deprel':
				factored_deptree = vocab.factorized
			elif vocab.field == 'semrel':
				factored_semgraph = vocab.factorized
		input_network_outputs = {}
		input_network_savers = []
		input_network_paths = []
		for input_network in self.input_networks:
			with tf.variable_scope(input_network.classname, reuse=False):
				input_network_outputs[input_network.classname] = input_network.build_graph(reuse=True)[0]
			network_variables = set(tf.global_variables(scope=input_network.classname))
			non_save_variables = set(tf.get_collection('non_save_variables'))
			network_save_variables = network_variables - non_save_variables
			saver = tf.train.Saver(list(network_save_variables))
			input_network_savers.append(saver)
			input_network_paths.append(self._config(self, input_network.classname+'_dir'))
		#Build Semantic parsing graph
		
		with tf.variable_scope(self.classname, reuse=False):
			train_graph = self.build_graph(input_network_outputs=input_network_outputs, reuse=False, debug=debug, nornn=nornn)
			train_outputs = TrainOutputs(*train_graph, load=load, evals=self._evals, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)
		with tf.variable_scope(self.classname, reuse=True):
			with tf.device('/device:GPU:0'):
				dev_graph = self.build_graph(input_network_outputs=input_network_outputs, reuse=True, debug=debug, nornn=nornn)
				dev_outputs = DevOutputs(*dev_graph, load=load, evals=self._evals, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)
		regularization_loss = self.l2_reg * tf.losses.get_regularization_loss() if self.l2_reg else 0

		#pdb.set_trace()
		update_step = tf.assign_add(self.global_step, 1)
		adam = AdamOptimizer(config=self._config)
		adam_op = adam.minimize(train_outputs.loss + regularization_loss, variables=tf.trainable_variables(scope=self.classname)) # returns the current step
		#adam_train_tensors = [adam_op, train_outputs.accuracies]
		adam_train_tensors = [adam_op, train_outputs.accuracies, train_outputs.get_print_dict]
		amsgrad = AMSGradOptimizer.from_optimizer(adam)
		amsgrad_op = amsgrad.minimize(train_outputs.loss + regularization_loss, variables=tf.trainable_variables(scope=self.classname)) # returns the current step
		#amsgrad_train_tensors = [amsgrad_op, train_outputs.accuracies]
		amsgrad_train_tensors = [amsgrad_op, train_outputs.accuracies, train_outputs.get_print_dict]
		if self.use_adamW:
			learning_rate = tf.train.exponential_decay(adam.learning_rate,self.global_step,adam.decay_steps,adam.decay_rate)
			adamw = tf.contrib.opt.AdamWOptimizer(self.l2_reg, learning_rate, adam.mu, adam.nu, adam.epsilon)
			#sgd = tf.train.AdamOptimizer(learning_rate,adam.mu,adam.nu)
			gvs = adamw.compute_gradients(train_outputs.loss, var_list=tf.trainable_variables(scope=self.classname))
			gradients, variables = zip(*gvs)
			gradients, _ = tf.clip_by_global_norm(gradients, adam.clip)
			adamw_op = adamw.apply_gradients(zip(gradients, variables))
			#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
			adamw_train_tensors = [adamw_op, train_outputs.accuracies, train_outputs.get_print_dict]
		#-----------------------------
		#use optimizer in tensorflow
		#pdb.set_trace()
		if self.use_sgd_loss:
			learning_rate = tf.train.exponential_decay(adam.learning_rate,self.global_step,adam.decay_steps,adam.decay_rate)
			sgd = tf.train.MomentumOptimizer(learning_rate, adam.momentum,use_nesterov=adam.nestrov)
			#sgd = tf.train.AdamOptimizer(learning_rate,adam.mu,adam.nu)
			gvs = sgd.compute_gradients(train_outputs.loss + regularization_loss, var_list=tf.trainable_variables(scope=self.classname))
			gradients, variables = zip(*gvs)
			gradients, _ = tf.clip_by_global_norm(gradients, adam.clip)
			sgd_op = sgd.apply_gradients(zip(gradients, variables))
			#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
			sgd_train_tensors = [sgd_op, train_outputs.accuracies, train_outputs.get_print_dict]

		dev_tensors = [dev_outputs.accuracies, dev_outputs.get_print_dict]
		# I think this needs to come after the optimizers
		if self.save_model_after_improvement or self.save_model_after_training:
			all_variables = set(tf.global_variables(scope=self.classname))
			non_save_variables = set(tf.get_collection('non_save_variables'))
			save_variables = all_variables - non_save_variables
			saver = tf.train.Saver(list(save_variables), max_to_keep=1)

		screen_output = []
		config = tf.ConfigProto()
		#Here is avoiding the out of memory, but I need to reduce the GPU usage later
		config.gpu_options.allow_growth = True
		#config.gpu_options.allow_growth = False
		#config.gpu_options.per_process_gpu_memory_fraction = 0.95
		config.allow_soft_placement = True
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			#BUG! save.restore should after global initializer
			for saver, path in zip(input_network_savers, input_network_paths):
				saver.restore(sess, path)
			
			if self.use_bert and not self.pretrained_bert:
				self.bertvocab.modelInit(sess)


			##---
			#os.makedirs(os.path.join(self.save_dir, 'profile'))
			#options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			#run_metadata = tf.RunMetadata()
			##---
			if not noscreen:
				#---------------------------------------------------------
				def run(stdscr):
					current_optimizer = 'Adam'
					train_tensors = adam_train_tensors
					if self.use_sgd_loss and not(self.switch_optimizers):
						current_optimizer = 'SGD'
						train_tensors = sgd_train_tensors
					if self.use_adamW:
						print('use adamw optimizer')
						current_optimizer = 'AdamW'
						train_tensors = adamw_train_tensors
					current_step = 0
					curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
					curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
					curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
					curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
					curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)
					curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)
					curses.init_pair(7, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
					stdscr.clrtoeol()
					stdscr.addstr('\t')
					stdscr.addstr('{}\n'.format(self.save_dir), curses.A_STANDOUT)
					stdscr.clrtoeol()
					stdscr.addstr('\t')
					stdscr.addstr('GPU: {}\n'.format(self.cuda_visible_devices), curses.color_pair(1) | curses.A_BOLD)
					stdscr.clrtoeol()
					stdscr.addstr('\t')
					stdscr.addstr('Current optimizer: {}\n'.format(current_optimizer), curses.color_pair(1) | curses.A_BOLD)
					stdscr.clrtoeol()
					stdscr.addstr('\t')
					stdscr.addstr('Epoch: {:3d}'.format(0), curses.color_pair(1) | curses.A_BOLD)
					stdscr.addstr(' | ')
					stdscr.addstr('Step: {:5d}\n'.format(0), curses.color_pair(1) | curses.A_BOLD)
					stdscr.clrtoeol()
					stdscr.addstr('\t')
					stdscr.addstr('Moving acc: {:5.2f}'.format(0.), curses.color_pair(1) | curses.A_BOLD)
					stdscr.addstr(' | ')
					stdscr.addstr('Best moving acc: {:5.2f}\n'.format(0.), curses.color_pair(1) | curses.A_BOLD)
					stdscr.clrtoeol()
					stdscr.addstr('\t')
					stdscr.addstr('Steps since improvement: {:4d}\n'.format(0),  curses.color_pair(1) | curses.A_BOLD)
					stdscr.clrtoeol()
					stdscr.move(2,0)
					stdscr.refresh()
					try:
						current_epoch = 0
						best_accuracy = 0
						current_accuracy = 0
						steps_since_best = 0
						while (not self.max_steps or current_step < self.max_steps) and \
									(not self.max_steps_without_improvement or steps_since_best < self.max_steps_without_improvement) and \
									(not self.n_passes or current_epoch < len(trainset.conllu_files)*self.n_passes):
							if steps_since_best >= self.switch_iter and self.switch_optimizers and (current_optimizer != 'AMSGrad' and current_optimizer != 'SGD'):
								if self.use_sgd_loss:
									current_optimizer = 'SGD'
									train_tensors = sgd_train_tensors
								else:
									train_tensors = amsgrad_train_tensors
									current_optimizer = 'AMSGrad'
							for batch in trainset.batch_iterator(shuffle=True):
								train_outputs.restart_timer()
								start_time = time.time()
								feed_dict = trainset.set_placeholders(batch)
								_, train_scores, printdata = sess.run(train_tensors, feed_dict=feed_dict)
								train_outputs.update_history(train_scores)
								current_step += 1
								if current_step % self.print_every == 0:
									for batch in devset.batch_iterator(shuffle=False):
										dev_outputs.restart_timer()
										feed_dict = devset.set_placeholders(batch)
										dev_scores, devprint = sess.run(dev_tensors, feed_dict=feed_dict)
										dev_outputs.update_history(dev_scores)

									#current_accuracy *= .5
									#current_accuracy += .5*dev_outputs.get_current_accuracy()
									current_accuracy = dev_outputs.get_current_accuracy()
									if current_accuracy >= best_accuracy:
										steps_since_best = 0
										best_accuracy = current_accuracy
										if self.save_model_after_improvement:
											saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
										if self.parse_devset:
											self.parse_files(devset, dev_outputs, sess, print_time=False)
									else:
										steps_since_best += self.print_every
									current_epoch = sess.run(self.global_step)
									stdscr.addstr('\t')
									stdscr.addstr('Current optimizer: {}\n'.format(current_optimizer), curses.color_pair(1) | curses.A_BOLD)
									stdscr.clrtoeol()
									stdscr.addstr('\t')
									stdscr.addstr('Epoch: {:3d}'.format(int(current_epoch)), curses.color_pair(1) | curses.A_BOLD)
									stdscr.addstr(' | ')
									stdscr.addstr('Step: {:5d}\n'.format(int(current_step)), curses.color_pair(1) | curses.A_BOLD)
									stdscr.clrtoeol()
									stdscr.addstr('\t')
									stdscr.addstr('Moving acc: {:5.2f}'.format(current_accuracy), curses.color_pair(1) | curses.A_BOLD)
									stdscr.addstr(' | ')
									stdscr.addstr('Best moving acc: {:5.2f}\n'.format(best_accuracy), curses.color_pair(1) | curses.A_BOLD)
									stdscr.clrtoeol()
									stdscr.addstr('\t')
									stdscr.addstr('Steps since improvement: {:4d}\n'.format(int(steps_since_best)),  curses.color_pair(1) | curses.A_BOLD)
									stdscr.clrtoeol()
									train_outputs.print_recent_history(stdscr)
									dev_outputs.print_recent_history(stdscr)
									print('')
									stdscr.move(2,0)
									stdscr.refresh()
							current_epoch = sess.run(self.global_step)
							sess.run(update_step)
							trainset.load_next()
						with open(os.path.join(self.save_dir, 'SUCCESS'), 'w') as f:
							pass
					except KeyboardInterrupt:
						pass

					line = 0
					stdscr.move(line,0)
					instr = stdscr.instr().rstrip()
					while instr:
						screen_output.append(instr)
						line += 1
						stdscr.move(line,0)
						instr = stdscr.instr().rstrip()
				#---------------------------------------------------------
				curses.wrapper(run)

				with open(os.path.join(self.save_dir, 'scores.txt'), 'wb') as f:
					#f.write(b'\n'.join(screen_output).decode('utf-8'))
					f.write(b'\n'.join(screen_output))
				print(b'\n'.join(screen_output).decode('utf-8'))
			else:
				current_optimizer = 'Adam'
				train_tensors = adam_train_tensors
				if self.use_sgd_loss and not(self.switch_optimizers):
					current_optimizer = 'SGD'
					train_tensors = sgd_train_tensors
				if self.use_adamW:
					print('use adamw optimizer')
					current_optimizer = 'AdamW'
					train_tensors = adamw_train_tensors
				current_step = 0
				print('\t', end='')
				print('{}\n'.format(self.save_dir), end='')
				print('\t', end='')
				print('GPU: {}\n'.format(self.cuda_visible_devices), end='')
				try:
					current_epoch = 0
					best_accuracy = 0
					current_accuracy = 0
					steps_since_best = 0
					decay_rate = 0
					bad_iteration_count = 0
					best_loss=9999999999999
					if self.classname=='ParserNetwork':
						class_target='deptree'
					elif self.classname=='GraphParserNetwork':
						class_target='semgraph'
					else:
						class_target=''
					while (not self.max_steps or current_step < self.max_steps) and \
								(not self.max_steps_without_improvement or steps_since_best < self.max_steps_without_improvement) and \
								(not self.n_passes or current_epoch < len(trainset.conllu_files)*self.n_passes):
						if steps_since_best >= self.switch_iter and self.switch_optimizers and (current_optimizer != 'AMSGrad' and current_optimizer != 'SGD'):
							if self.use_sgd_loss:
								current_optimizer = 'SGD'
								train_tensors = sgd_train_tensors
							else:
								train_tensors = amsgrad_train_tensors
								current_optimizer = 'AMSGrad'
							print('\t', end='')
							print('Current optimizer: {}\n'.format(current_optimizer), end='')
						for batch in trainset.batch_iterator(shuffle=False):
							train_outputs.restart_timer()
							start_time = time.time()
							feed_dict = trainset.set_placeholders(batch)
							if debug:
								test=list(feed_dict.keys())
								# pdb.set_trace()
								
							# if debug:
							#   sess=tf_debug.LocalCLIDebugWrapperSession(sess)
							# pdb.set_trace()
							# exit()
							##---
							#if current_step < 10:
							#  _, train_scores = sess.run(train_tensors, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
							#  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
							#  chrome_trace = fetched_timeline.generate_chrome_trace_format()
							#  with open(os.path.join(self.save_dir, 'profile', 'timeline_step_%d.json' % current_step), 'w') as f:
							#    f.write(chrome_trace)
							#else:
							#  _, train_scores = sess.run(train_tensors, feed_dict=feed_dict)
							#run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
							run_options = tf.RunOptions()
							_, train_scores, printdata = sess.run(train_tensors, feed_dict=feed_dict, options=run_options)
							
							if debug:
								# pdb.set_trace()
								pass
							##---
							train_outputs.update_history(train_scores)
							current_step += 1
							if current_step % self.print_every == 0:
								if current_optimizer=='Adam':
									current_optm=adam
								elif current_optimizer=='AMSGrad':
									current_optm=amsgrad

								if debug:
									# pdb.set_trace()
									pass
								if self.dev_mst:
									# pdb.set_trace()
									self.parse_file(devset, dev_outputs, sess, output_dir='results', output_filename='dev_temp_'+str(uid)+'.conllu', print_time=False)
									correct=self.evaluate('results/dev_temp_'+str(uid)+'.conllu', devset._conllu_files[0])
									correct_punct=self.evaluate('results/dev_temp_'+str(uid)+'.conllu', devset._conllu_files[0], punct=[])
									if self.average_AS:
										current_accuracy=(correct['LAS']+correct['UAS'])/2
									else:
										current_accuracy=correct['LAS']
								else:
									for batch in devset.batch_iterator(shuffle=False):
										dev_outputs.restart_timer()
										feed_dict = devset.set_placeholders(batch)
										try:
											dev_scores, devprint = sess.run(dev_tensors, feed_dict=feed_dict)
										except:
											pdb.set_trace()
										# pdb.set_trace()
										dev_outputs.update_history(dev_scores)
									#current_accuracy *= .5
									#current_accuracy += .5*dev_outputs.get_current_accuracy()
									current_accuracy = dev_outputs.get_current_accuracy()
								if self.dev_mst:
									if self.average_AS:
										target_bool=current_accuracy >= best_accuracy
									else:
										target_bool=current_accuracy > best_accuracy or (current_accuracy==best_accuracy and correct['UAS'] > best['UAS'])
									if target_bool:
										best_punct=correct_punct
										best=correct
										steps_since_best = 0
										best_accuracy = current_accuracy
										if self.save_model_after_improvement:
											saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
											if self.use_bert:
												if self.bertvocab.is_training:
													self.bertvocab.modelSave(sess,os.path.join(self.save_dir,'bert'),self.global_step)
										if self.parse_devset:
											self.parse_files(devset, dev_outputs, sess, print_time=False)
									else:
										steps_since_best += self.print_every
								elif current_accuracy >= best_accuracy:
									steps_since_best = 0
									best_accuracy = current_accuracy
									if self.save_model_after_improvement:
										saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
										if self.use_bert:
											if self.bertvocab.is_training:
												self.bertvocab.modelSave(sess,os.path.join(self.save_dir,'bert'),self.global_step)
									if self.parse_devset:
										self.parse_files(devset, dev_outputs, sess, print_time=False)
								else:
									steps_since_best += self.print_every


								if self.loss_based_decay_schedule:
									current_loss=train_outputs.history[class_target]['loss'][-1]/train_outputs.history['total']['n_batches']
									if current_loss<best_loss:
										best_loss=current_loss
										bad_iteration_count=0
									else:
										bad_iteration_count+=self.print_every
									if bad_iteration_count>=self.decay_steps:
										bad_iteration_count=0
										sess.run(tf.assign_add(current_optm.decay_counts,1))
								elif self.improvement_based_decay_schedule:
									if steps_since_best>0:
										bad_iteration_count+=self.print_every
									else:
										bad_iteration_count=0
									if bad_iteration_count>=self.decay_steps:
										sess.run(tf.assign_add(current_optm.decay_counts,1))
										bad_iteration_count=0

								if debug:
									pdb.set_trace()
									pass

								current_epoch = sess.run(self.global_step)
								#pdb.set_trace()
								if current_optimizer=='SGD':
									print('Current LR: {:3f}'.format(float(sess.run(learning_rate))))
								elif current_optimizer=='Adam':
									print('Current LR: {:3f}'.format(float(sess.run(adam.annealed_learning_rate))))
								elif current_optimizer=='AMSGrad':
									print('Current LR: {:3f}'.format(float(sess.run(amsgrad.annealed_learning_rate))))
								if self.loss_based_decay_schedule:
									# pdb.set_trace()
									print('Current decay count: {:d}'.format(int(sess.run(current_optm.decay_counts))))
								print('\t', end='')
								print('Epoch: {:3d}'.format(int(current_epoch)), end='')
								print(' | ', end='')
								print('Step: {:5d}\n'.format(int(current_step)), end='')
								print('\t', end='')
								print('Moving acc: {:5.2f}'.format(current_accuracy), end='')
								print(' | ', end='')
								print('Best moving acc: {:5.2f}\n'.format(best_accuracy), end='')
								print('\t', end='')
								print('Steps since improvement: {:4d}\n'.format(int(steps_since_best)), end='')
								train_outputs.print_recent_history()
								if self.dev_mst:
									print('Current dev with punctuations: UAS: {:5.2f}, LAS: {:5.2f}'.format(correct_punct['UAS'],correct_punct['LAS']))
									print('Best dev with punctuations: UAS: {:5.2f}, LAS: {:5.2f}'.format(best_punct['UAS'],best_punct['LAS']))
									print('Current dev no punctuations: UAS: {:5.2f}, LAS: {:5.2f}'.format(correct['UAS'],correct['LAS']))
									print('Best dev no punctuations: UAS: {:5.2f}, LAS: {:5.2f}'.format(best['UAS'],best['LAS']))

								else:
									dev_outputs.print_recent_history()
						current_epoch = sess.run(self.global_step)
						sess.run(update_step)
						trainset.load_next()
					with open(os.path.join(self.save_dir, 'SUCCESS'), 'w') as f:
						pass
				except KeyboardInterrupt:
					pass
			if self.save_model_after_training:
				saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
		return
	def evaluate(self, filename, target_filename, punct=['.', '``', "''", ':', ',']):
		""""""
		
		correct = {'UAS': [], 'LAS': []}
		with open(target_filename) as f_tar:
			with open(filename) as f:
				for line in f:
					line_tar = f_tar.readline()
					line = line.strip().split('\t')
					line_tar = line_tar.strip().split('\t')
					if len(line) == 10 and line[4] not in punct:
						correct['UAS'].append(0)
						correct['LAS'].append(0)
						try:
							assert line_tar[1]==line[1], "two files are not equal!"
						except:
							pdb.set_trace()
						if line[6] == line_tar[6]:
							correct['UAS'][-1] = 1
							if line[7] == line_tar[7]:
								correct['LAS'][-1] = 1
		#correct = {k:np.array(v) for k, v in correct.iteritems()}
		correct['UAS']=np.mean(correct['UAS']) * 100
		correct['LAS']=np.mean(correct['LAS']) * 100
		return correct
		#return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct
	#=============================================================
	def parse(self, conllu_files, output_dir=None, output_filename=None, testing=False, debug=False,nornn=False,check_iter=False, gen_tree=False, get_argmax=False):
		""""""

		if testing:
			#save_dir=self.save_dir
			#print(save_dir.split('GraphParserNetwork')[-1],end='\t')
			blockPrint()
			pass
		else:
			blockPrint()
			pass
		parseset = conllu_dataset.CoNLLUDataset(conllu_files, self.vocabs,
																						config=self._config)

		if output_filename:
			assert len(conllu_files) == 1, "output_filename can only be specified for one input file"
		factored_deptree = None
		factored_semgraph = None
		for vocab in self.output_vocabs:
			if vocab.field == 'deprel':
				factored_deptree = vocab.factorized
			elif vocab.field == 'semrel':
				factored_semgraph = vocab.factorized
		compare_two=False
		#pdb.set_trace()
		if compare_two:
			with tf.variable_scope(self.classname, reuse=False):
				parse_graph = self.build_graph(reuse=True, debug=debug, nornn=nornn)
				parse_outputs = DevOutputs(*parse_graph, load=False, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)
			parse_tensors = parse_outputs.accuracies
			all_variables = set(tf.global_variables())
			non_save_variables = set(tf.get_collection('non_save_variables'))
			save_variables = all_variables - non_save_variables
			saver = tf.train.Saver(list(save_variables), max_to_keep=1)
			with tf.variable_scope('testing',reuse=False):
				parse_graph2 = self.build_graph(reuse=True, debug=debug, nornn=nornn)
				parse_outputs2 = DevOutputs(*parse_graph2, load=False, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)
			parse_tensors2 = parse_outputs2.accuracies
			all_variables2 = set(tf.global_variables())
			non_save_variables2 = set(tf.get_collection('non_save_variables'))
			save_variables2 = all_variables2 - non_save_variables2
			saver2 = tf.train.Saver(list(save_variables2), max_to_keep=1)
			sess.run(tf.variables_initializer(list(non_save_variables)))
			sess.run(tf.variables_initializer(list(non_save_variables)))
			saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
			saver2.restore(sess, tf.train.latest_checkpoint(self.model_dir))
		else:
			with tf.variable_scope(self.classname, reuse=False):
					parse_graph = self.build_graph(reuse=True, debug=debug, nornn=nornn)
					parse_outputs = DevOutputs(*parse_graph, load=False, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)
			#pdb.set_trace()
			parse_tensors = parse_outputs.accuracies
			all_variables = set(tf.global_variables())
			non_save_variables = set(tf.get_collection('non_save_variables'))
			if self.use_bert:
				bert_variables=set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='bert'))
			else:
				bert_variables=set()
			save_variables = all_variables - non_save_variables - bert_variables
			saver = tf.train.Saver(list(save_variables), max_to_keep=1)
		if testing:
			enablePrint()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		with tf.Session(config=config) as sess:
			sess.run(tf.variables_initializer(list(non_save_variables)))
			#pdb.set_trace()
			if testing:
				saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
				
				'''
				Label_list=[var for var in save_variables if '/Labeled/' in str(var)]
				saver2 = tf.train.Saver(Label_list, max_to_keep=1)
				saver2.restore(sess, tf.train.latest_checkpoint('saves/SemEval15/DM_modified/GraphParserNetwork1iter_600unary_200hidden_batch6000_sep_embed_dm_001_subtoken_1init_LBP'))
				'''
			else:
				saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
			if self.use_bert and not self.pretrained_bert:
				if self.bertvocab.is_training:
					self.bertvocab.modelRestore(sess,list(bert_variables),model_dir=os.path.join(self.model_dir,'bert'))
				else:
					self.bertvocab.modelRestore(sess,list(bert_variables))

			parse_outputs.id_buff=parseset.id_buff
			if testing:
				if not gen_tree:
					self.parse_score(parseset, parse_outputs, sess, output_dir=output_dir, output_filename=output_filename, debug=debug, check_iter=check_iter)
					if get_argmax:
						self.parse_file(parseset, parse_outputs, sess, output_dir=output_dir, output_filename=output_filename,get_argmax=get_argmax)
				else:
					#'''
					self.temp_vocabs = []
					for output_vocab_classname in ['DepheadIndexVocab','DeprelTokenVocab']:
						VocabClass = getattr(vocabs, output_vocab_classname)
						vocab = VocabClass(config=self._config)
						vocab.load() or vocab.count(self.train_conllus)
						self.temp_vocabs.append(vocab)
					self.parse_MST(parseset, parse_outputs, sess, output_dir=output_dir, output_filename=output_filename)
					#'''
			else:
					if len(conllu_files) == 1 or output_filename is not None:
						self.parse_file(parseset, parse_outputs, sess, output_dir=output_dir, output_filename=output_filename)
					else:
						self.parse_files(parseset, parse_outputs, sess, output_dir=output_dir)
		
		'''
		parse_scores = sess.run(parse_tensors, feed_dict=feed_dict)
		parse_outputs.update_history(parse_scores)
		parse_outputs.print_recent_history()
		dev_outputs.accuracies
		'''
		return
	def parse_score(self, dataset, graph_outputs, sess, output_dir=None, output_filename=None, print_time=True, debug=False, check_iter=False):
		#if debug:
		probability_tensors = [graph_outputs.accuracies,graph_outputs.get_print_dict]
		#else:
		#    probability_tensors = graph_outputs.accuracies
		#pdb.set_trace()
		input_filename = dataset.conllu_files[0]
		graph_outputs.restart_timer()
		#pdb.set_trace()
		if debug:
				#how many second order add edges
				so_modified_count=0
				total_sent=0
				#second order makes better
				so_modified_good_count=0
				#second order makes worse
				so_modified_bad_count=0
				so_modified_count_temp=0
				so_modified_good_count_temp=0
				so_modified_bad_count_temp=0
		for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
			graph_outputs.restart_timer()
			tokens, lengths = dataset.get_tokens(indices)
			feed_dict = dataset.set_placeholders(indices)
			scores, devprint = sess.run(probability_tensors, feed_dict=feed_dict)
			graph_outputs.update_history(scores)
			#pdb.set_trace()
			clause=['(' in sent for sent in tokens['form']]
			#'''
			if debug:
				total_sent+=devprint['printdata']['targets'].shape[0]
				targets=devprint['printdata']['targets'].transpose([0,2,1])
				weights=devprint['printdata']['token_weights'].transpose([0,2,1])
				#pdb.set_trace()
				if self.model_type=='CRF':
					#second_predict=np.argmax(devprint['printdata']['q_value'],2)*weights
					#unary_predict=np.argmax(-devprint['printdata']['unary'],2)*weights
					#pdb.set_trace()
					second_predict=np.argmax(devprint['printdata']['q_value'],2)*weights
					unary_predict=np.greater(devprint['printdata']['unary'],0)*weights
					if check_iter:
						q0=np.argmax(devprint['printdata']['q_value0'],2)*weights
						q1=np.argmax(devprint['printdata']['q_value1'],2)*weights
						q2=np.argmax(devprint['printdata']['q_value2'],2)*weights
						q_orig=np.argmax(devprint['q_value_old'],2)*weights
						change_state=(np.abs(q2-q1).sum(-1).sum(-1)>0).astype(int)+(np.abs(q1-q0).sum(-1).sum(-1)>0).astype(int)+(np.abs(q0-q_orig).sum(-1).sum(-1)>0).astype(int)+(np.abs(q2-q0).sum(-1).sum(-1)>0).astype(int)+(np.abs(q1-q_orig).sum(-1).sum(-1)>0).astype(int)
						if (change_state>=5).sum()>0:
							
							changeidx=np.where(change_state==5)[0]
							sentenceids=np.array(dataset.id_buff)
							currentids=sentenceids[indices]
							target_sent=currentids[changeidx]
							q_orig_change=q_orig[changeidx]
							q0change=q0[changeidx]
							q1change=q1[changeidx]
							q2change=q2[changeidx]
							#qval_change=second_predict[changeidx]
							target_graph=targets[changeidx]
							vj=0
							if (target_graph==q2change).all() and q2change.shape[1]<9:
								print(target_sent)
								#continue
								# pdb.set_trace()
								'''
								np.set_printoptions(precision=2,linewidth=200)
								vj=1
								tgt=1
								lng=5
								iter=2
								unary=devprint['printdata']['unary'][changeidx[tgt]][:lng,:lng]
								mess_sib=np.einsum('ab,abc->abc',devprint['q_value'+str(iter)][changeidx[tgt]][:,1,:],devprint['printdata']['layer_sib'][changeidx[tgt]])[:lng,:lng,:lng]
								mess_cop=np.einsum('ab,abc->abc',devprint['q_value'+str(iter)][changeidx[tgt]][:,1,:],devprint['printdata']['layer_cop'][changeidx[tgt]])[:lng,:lng,:lng]
								mess_gp1=np.einsum('ab,abc->abc',devprint['q_value'+str(iter)][changeidx[tgt]][:,1,:],devprint['printdata']['layer_gp'][changeidx[tgt]])[:lng,:lng,:lng]
								mess_gp2=np.einsum('ab,abc->abc',devprint['q_value'+str(iter)][changeidx[tgt]][:,1,:],devprint['printdata']['layer_gp'][changeidx[tgt]].transpose([1,2,0]))[:lng,:lng,:lng]
								#ab->ac / [a,1,c]
								mess_sib=mess_sib/np.abs(unary[:,None,:]+1e-12)
								#ab->cb / [1,c,b]
								mess_cop=mess_cop/np.abs(unary.T[None,:,:]+1e-12)
								#ab->bc / [1,b,c]
								mess_gp1=mess_gp1/np.abs(unary[None,:,:]+1e-12)
								#ab->ca / [c,1,a]
								mess_gp2=mess_gp2/np.abs(unary.T[:,None,:]+1e-12)
								mess_sib=self.message_process(mess_sib)
								mess_cop=self.message_process(mess_cop)
								mess_gp1=self.message_process(mess_gp1)
								mess_gp2=self.message_process(mess_gp2)
								mess_sib,mess_cop,mess_gp1,mess_gp2=sess.run([mess_sib,mess_cop,mess_gp1,mess_gp2])
								self.write_latex_message(mess_sib,type='sib')
								self.write_latex_message(mess_cop,type='cop')
								self.write_latex_message(mess_gp1,type='gp1')
								self.write_latex_message(mess_gp2,type='gp2')
								'''
								#'''
								print(target_sent)
								print(q_orig_change[vj])
								print(q0change[vj])
								print(q1change[vj])
								print(q2change[vj])
								print(target_graph[vj])
								print(tokens['form'][changeidx[vj]])
								print(tokens['semrel'][changeidx[vj]])
								print(devprint['printdata']['second_temp0'][changeidx[vj]])
								print(devprint['printdata']['second_temp1'][changeidx[vj]])
								print(devprint['printdata']['second_temp2'][changeidx[vj]])
								print(devprint['printdata']['second_temp_sib0after'][changeidx[vj]])
								print(devprint['printdata']['second_temp_sib1after'][changeidx[vj]])
								print(devprint['printdata']['second_temp_sib2after'][changeidx[vj]])
								#'''
					temp_predict=np.argmax(devprint['printdata']['q_value0'],2)*weights
					modified_temp=abs(temp_predict-unary_predict)
					((targets==temp_predict)*modified_temp).sum()
					so_modified_count_temp+=modified_temp.sum()
					so_modified_good_count_temp+=((targets==temp_predict)*modified_temp).sum()
					so_modified_bad_count_temp+=((targets!=temp_predict)*modified_temp).sum()
				else:
					second_predict=np.argmax(devprint['printdata']['q_value'],1)*weights
					unary_predict=np.argmax(devprint['printdata']['unary'],1)*weights
				modified=abs(second_predict-unary_predict)
				((targets==second_predict)*modified).sum()
				so_modified_count+=modified.sum()
				so_modified_good_count+=((targets==second_predict)*modified).sum()
				so_modified_bad_count+=((targets!=second_predict)*modified).sum()
				#so_modified_bad_count+=((targets!=second_predict)*modified).sum()
				#'''
				#pdb.set_trace()
		#pdb.set_trace()
		if debug:
			#print('TP/FPFN:',so_modified_good_count,'/',so_modified_bad_count,end='\t')
			print(so_modified_good_count,'/',so_modified_bad_count,end=' ',sep='')
			#print('FPFN:',so_modified_bad_count,end='\t')
			#print('TP0/FPFN0:',so_modified_good_count_temp,'/',so_modified_bad_count_temp,end='\t')
			print(so_modified_good_count_temp,'/',so_modified_bad_count_temp,end=' ',sep='')
			#print('FPFN0:',so_modified_bad_count_temp,end='\t')
		#precision = graph_outputs.history['semgraph']['tokens'][-1] / (graph_outputs.history['semgraph']['tokens'][-1] + graph_outputs.history['semgraph']['fp_tokens'] + 1e-12)
		#recall = graph_outputs.history['semgraph']['tokens'][-1] / (graph_outputs.history['semgraph']['tokens'][-1] + graph_outputs.history['semgraph']['fn_tokens'] + 1e-12)
		#print(2 * (precision * recall) / (precision + recall + 1e-12))
		graph_outputs.print_recent_history(dataprint=True)
		#current_accuracy = .5*graph_outputs.get_current_accuracy()
		#if print_time:
		#  print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - graph_outputs.time))
		return
	#=============================================================
	def parse_MST(self, dataset, graph_outputs, sess, output_dir=None, output_filename=None, print_time=False):
		""""""

		probability_tensors = graph_outputs.probabilities
		input_filename = dataset.conllu_files[0]
		graph_outputs.restart_timer()
		for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
			tokens, lengths = dataset.get_tokens(indices)
			feed_dict = dataset.set_placeholders(indices)
			probabilities = sess.run(probability_tensors, feed_dict=feed_dict)
			#pdb.set_trace()
			predictions = graph_outputs.probs_to_preds(probabilities, lengths, force_MST=True)
			tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.temp_vocabs})
			graph_outputs.cache_predictions(tokens, indices)

		if output_dir is None and output_filename is None:
			graph_outputs.print_current_predictions()
		else:
			input_dir, input_filename = os.path.split(input_filename)
			if output_dir is None:
				output_dir = os.path.join(self.save_dir, 'parsed', input_dir)
			elif output_filename is None:
				output_filename = input_filename
			
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
			output_filename = os.path.join(output_dir, output_filename)
			with codecs.open(output_filename, 'w', encoding='utf-8') as f:
				graph_outputs.dump_current_predictions(f)
		if print_time:
			print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - graph_outputs.time))
		return
	#=============================================================
	def parse_file(self, dataset, graph_outputs, sess, output_dir=None, output_filename=None, print_time=False,get_argmax=False):
		""""""

		probability_tensors = graph_outputs.probabilities
		input_filename = dataset.conllu_files[0]
		graph_outputs.restart_timer()
		for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
			tokens, lengths = dataset.get_tokens(indices)
			feed_dict = dataset.set_placeholders(indices)
			probabilities = sess.run(probability_tensors, feed_dict=feed_dict)
			predictions = graph_outputs.probs_to_preds(probabilities, lengths, get_argmax=get_argmax)
			try:
				tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.output_vocabs})
			except:
				pdb.set_trace()
			graph_outputs.cache_predictions(tokens, indices)

		if output_dir is None and output_filename is None:
			graph_outputs.print_current_predictions()
		else:
			input_dir, input_filename = os.path.split(input_filename)
			if output_dir is None:
				output_dir = os.path.join(self.save_dir, 'parsed', input_dir)
			elif output_filename is None:
				output_filename = input_filename
			
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
			output_filename = os.path.join(output_dir, output_filename)
			with codecs.open(output_filename, 'w', encoding='utf-8') as f:
				graph_outputs.dump_current_predictions(f)
		if print_time:
			print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - graph_outputs.time))
		return

	#=============================================================
	def parse_files(self, dataset, graph_outputs, sess, output_dir=None, print_time=True):
		""""""

		probability_tensors = graph_outputs.probabilities
		graph_outputs.restart_timer()
		for input_filename in dataset.conllu_files:
			for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
				tokens, lengths = dataset.get_tokens(indices)
				feed_dict = dataset.set_placeholders(indices)
				probabilities = sess.run(probability_tensors, feed_dict=feed_dict)
				predictions = graph_outputs.probs_to_preds(probabilities, lengths)
				tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.output_vocabs})
				graph_outputs.cache_predictions(tokens, indices)

			input_dir, input_filename = os.path.split(input_filename)
			if output_dir is None:
				file_output_dir = os.path.join(self.save_dir, 'parsed', input_dir)
			else:
				file_output_dir = output_dir
			if not os.path.exists(file_output_dir):
				os.makedirs(file_output_dir)
			output_filename = os.path.join(file_output_dir, input_filename)
			with codecs.open(output_filename, 'w', encoding='utf-8') as f:
				graph_outputs.dump_current_predictions(f)
			
			# Load the next conllu file
			dataset.load_next()
		
		if print_time:
			n_files = len(dataset.conllu_files)
			print('\033[92mParsing {} file{} took {:0.1f} seconds\033[0m'.format(n_files, 's' if n_files > 1 else '', time.time() - graph_outputs.time))
		return

	#=============================================================
	def get_input_tensor(self, outputs, reuse=True):
		""""""

		output_keep_prob = 1. if reuse else self.output_keep_prob
		for output in outputs:
			pass # we just need to grab one
		layer = output['recur_layer']
		with tf.variable_scope(self.classname):
			layer = classifiers.hiddens(layer, self.output_size,
																	hidden_func=self.output_func,
																	hidden_keep_prob=output_keep_prob,
																	reuse=reuse)
		return [layer]
	#=============================================================
	def message_process(self, mess):
		#remove a b b
		mess-=tf.matrix_band_part(mess, 0, 0)
		#remove b a b
		mess-=tf.transpose(tf.matrix_band_part(tf.transpose(mess,[1,0,2]), 0, 0),[1,0,2])
		#remove b b a
		mess-=tf.transpose(tf.matrix_band_part(tf.transpose(mess,[2,0,1]), 0, 0),[1,2,0])
		return mess
	def write_latex_message(self, mess, type='sib',thres=1,opacity_fix=False,width_fix=False,fix_line=True,width_thres=3):
		#pdb.set_trace()
		indices=np.where(np.abs(mess)>thres)
		if not fix_line:
			basesent='\draw[->,line width={:.2f}mm,{:s},opacity={:.2f}] (G-{:d}{:d}) to[out={:.1f},in={:.1f}] (G-{:d}{:d});'
		else:
			basesent='\draw[->,line width={:.2f}mm,{:s},opacity={:.2f}] (G-{:d}{:d}) to[line] (G-{:d}{:d});'
		for i in range(indices[0].shape[0]):
			A=indices[0][i]+1
			B=indices[1][i]+1
			if type=='sib':
				#ab->ac
				C=indices[0][i]+1
				D=indices[2][i]+1
			if type=='cop':
				#ab->cb
				C=indices[2][i]+1
				D=indices[1][i]+1
			if type=='gp1':
				#ab->bc
				C=indices[1][i]+1
				D=indices[2][i]+1
			if type=='gp2':
				#ab->ca
				C=indices[2][i]+1
				D=indices[0][i]+1
			if mess[indices[0][i],indices[1][i],indices[2][i]] < 0:
				draw_color='red'
			else:
				draw_color='blue'
			draw_width=np.abs(mess[indices[0][i],indices[1][i],indices[2][i]])/width_thres
			if draw_width>=1:
				draw_width=1
			B-=1
			D-=1
			outang,inang=self.calc_angle(A,B,C,D,cross=self.check_node_cross(A,B,C,D))
			#pdb.set_trace()
			if not fix_line:
				if opacity_fix:
					print(basesent.format(draw_width,draw_color,1,A,B,outang,inang,C,D))
				elif width_fix:
					print(basesent.format(0.5,draw_color,draw_width,A,B,outang,inang,C,D))
				else:
					print(basesent.format(draw_width,draw_color,draw_width,A,B,outang,inang,C,D))
			else:
				if opacity_fix:
					print(basesent.format(draw_width,draw_color,1,A,B,C,D))
				elif width_fix:
					print(basesent.format(0.5,draw_color,draw_width,A,B,C,D))
				else:
					print(basesent.format(draw_width,draw_color,draw_width,A,B,C,D))
			#print('\draw[->,line width=1mm,red,opacity=0.5] (G-{:d}{:d}) to[out=120,in=-30] (G-{:d}{:d});'.format(A,B,C,D))
		return 
	def calc_angle(self,A,B,C,D,cross=False):
		if A==C:
			if B>D:
				outang=180
				inang=0
			else:
				outang=0
				inang=180
		elif B==D:
			if A>C:
				outang=90
				inang=-90
			else:
				outang=-90
				inang=90
		else:
			#angle=(A-C)/(B-D)*45
			angle=np.rad2deg(np.arctan((B-D)/(A-C)))
			if A>C:
				direction=1
			else:
				direction=-1
			outang=90*direction+angle
			inang=-90*direction+angle
		sel=[1,-1]
		if cross:
			newdir=(np.random.choice(sel,1)*15)[0]
			dir_size=15
			if A==1 and C==1:
				if B<D:
					newdir=-dir_size
				else:
					newdir=dir_size
			if A==6 and C==6:
				if B<D:
					newdir=+dir_size
				else:
					newdir=-dir_size
			if B==1 and D==1:
				if A<C:
					newdir=+dir_size
				else:
					newdir=-dir_size
			if B==6 and D==6:
				if A<C:
					newdir=-dir_size
				else:
					newdir=+dir_size
			if A!=C and B!=D:
				if np.abs(angle)<20 or np.abs(angle)>70:
					newdir=0
			outang+=newdir
			inang-=newdir
		return outang,inang
	def check_node_cross(self,A,B,C,D):
		if abs(A-C)>1 or abs(B-D)>1:
			return True
		return False
	#=============================================================
	@property
	def model_type(self):
		if self._config.has_section('SecondOrderVocab'):
			return 'CRF'
		else:
			return 'LBP'
	@property
	def train_conllus(self):
		return self._config.getfiles(self, 'train_conllus')
	@property
	def cuda_visible_devices(self):
		return os.getenv('CUDA_VISIBLE_DEVICES')
	@property
	def save_dir(self):
		'''
		if self._config.get('DEFAULT','AUTO_dir')=='True':
			return self._config.get('DEFAULT', 'save_dir')+self._config.get('DEFAULT','modelname')
		'''
		return self._config.getstr(self, 'save_dir')
	@property
	def model_dir(self):
		#if self._config.get('DEFAULT','AUTO_dir')=='True':
		return self._config.getstr(self, 'save_dir')
		'''
		try:
			return self._config.get('DEFAULT', 'save_dir')+self._config.get('DEFAULT','modelname')
		except:
			return self._config.getstr(self, 'save_dir')
		'''
	@property
	def vocabs(self):
		return self._vocabs
	@property
	def id_vocab(self):
		return self._id_vocab
	@property
	def node_id_vocab(self):
		return self._node_id_vocab
	@property
	def input_vocabs(self):
		return self._input_vocabs
	@property
	def decoder_vocabs(self):
		return self._decoder_vocabs
	@property
	def throughput_vocabs(self):
		return self._throughput_vocabs
	@property
	def output_vocabs(self):
		return self._output_vocabs
	@property
	def input_networks(self):
		return self._input_networks
	@property
	def input_network_classes(self):
		return self._config.getlist(self, 'input_network_classes')
	@property
	def input_vocab_classes(self):
		return self._config.getlist(self, 'input_vocab_classes')
	@property
	def output_vocab_classes(self):
		return self._config.getlist(self, 'output_vocab_classes')
	@property
	def throughput_vocab_classes(self):
		return self._config.getlist(self, 'throughput_vocab_classes')
	@property
	def l2_reg(self):
		return self._config.getfloat(self, 'l2_reg')
	@property
	def input_size(self):
		return self._config.getint(self, 'input_size')
	@property
	def recur_size(self):
		return self._config.getint(self, 'recur_size')
	@property
	def n_layers(self):
		return self._config.getint(self, 'n_layers')
	@property
	def first_layer_conv_width(self):
		return self._config.getint(self, 'first_layer_conv_width')
	@property
	def conv_width(self):
		return self._config.getint(self, 'conv_width')
	@property
	def input_keep_prob(self):
		return self._config.getfloat(self, 'input_keep_prob')
	@property
	def conv_keep_prob(self):
		return self._config.getfloat(self, 'conv_keep_prob')
	@property
	def recur_keep_prob(self):
		return self._config.getfloat(self, 'recur_keep_prob')
	@property
	def recur_include_prob(self):
		return self._config.getfloat(self, 'recur_include_prob')
	@property
	def bidirectional(self):
		return self._config.getboolean(self, 'bidirectional')
	@property
	def input_func(self):
		input_func = self._config.getstr(self, 'input_func')
		if hasattr(nonlin, input_func):
			return getattr(nonlin, input_func)
		else:
			raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, input_func))
	@property
	def hidden_func(self):
		hidden_func = self._config.getstr(self, 'hidden_func')
		if hasattr(nonlin, hidden_func):
			return getattr(nonlin, hidden_func)
		else:
			raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
	@property
	def recur_func(self):
		recur_func = self._config.getstr(self, 'recur_func')
		if hasattr(nonlin, recur_func):
			return getattr(nonlin, recur_func)
		else:
			raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, recur_func))
	@property
	def highway_func(self):
		highway_func = self._config.getstr(self, 'highway_func')
		if hasattr(nonlin, highway_func):
			return getattr(nonlin, highway_func)
		else:
			raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, highway_func))
	@property
	def recur_cell(self):
		recur_cell = self._config.getstr(self, 'recur_cell')
		if hasattr(recurrent, recur_cell):
			return getattr(recurrent, recur_cell)
		else:
			raise AttributeError("module '{}' has no attribute '{}'".format(recurrent.__name__, recur_cell))
	@property
	def cifg(self):
		return self._config.getboolean(self, 'cifg')
	@property
	def bilin(self):
		return self._config.getboolean(self, 'bilin')
	@property
	def switch_optimizers(self):
		return self._config.getboolean(self, 'switch_optimizers')
	@property
	def highway(self):
		return self._config.getboolean(self, 'highway')
	@property
	def print_every(self):
		return self._config.getint(self, 'print_every')
	@property
	def max_steps(self):
		return self._config.getint(self, 'max_steps')
	@property
	def max_steps_without_improvement(self):
		return self._config.getint(self, 'max_steps_without_improvement')
	@property
	def n_passes(self):
		return self._config.getint(self, 'n_passes')
	@property
	def parse_devset(self):
		return self._config.getboolean(self, 'parse_devset')
	@property
	def save_model_after_improvement(self):
		return self._config.getboolean(self, 'save_model_after_improvement')
	@property
	def save_model_after_training(self):
		return self._config.getboolean(self, 'save_model_after_training')
	@property
	def classname(self):
		return self.__class__.__name__
	@property
	def share_layer(self):
		return self._config.getboolean(self, 'share_layer')
	@property
	def use_sgd_loss(self):
		try:
			return self._config.getboolean(self, 'use_sgd_loss')
		except:
			return False
	@property
	def use_adamW(self):
		try:
			return self._config.getboolean(self, 'use_adamW')
		except:
			return False
	@property
	def switch_iter(self):
		try:
			return self._config.getint(self, 'switch_iter')
		except:
			return 500
	@property
	def get_nodes_path(self):
		return self._config.get('BaseNetwork', 'nodes_path')

	@property
	def dev_mst(self):
		try:
			return self._config.getboolean(self, 'dev_mst')
		except:
			return False
	@property
	def average_AS(self):
		try:
			return self._config.getboolean(self, 'average_as')
		except:
			return False
	@property
	def loss_based_decay_schedule(self):
		try:
			return self._config.get('Optimizer', 'loss_based_decay_schedule')=='True'
		except:
			return False
	@property
	def improvement_based_decay_schedule(self):
		try:
			return self._config.get('Optimizer', 'improvement_based_decay_schedule')=='True'
		except:
			return False
	@property
	def decay_steps(self):
		try:
			return int(self._config.get('Optimizer', 'decay_steps'))
		except:
			return False