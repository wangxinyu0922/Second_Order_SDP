# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb
import copy

import tensorflow as tf
import parser.layers as layers
from parser.neural import nn, nonlin, embeddings, classifiers, recurrent



def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)

def _decoder(cell1, cell2, inputs, memory, sequence_length, initial_state, dtype=None,
             scope=None, target_copy_hidden_states=None):
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell2.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)
    target_copy_attentions = []
    # pdb.set_trace()
    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])
        mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(memory)[1],
                                    dtype=dtype)
        bias = layers.attention.attention_bias(mem_mask, "masking",
                                               dtype=dtype)
        bias = tf.squeeze(bias, axis=[1, 2])
        cache = layers.attention.attention(None, memory, None, output_size)

        # bias2 = layers.attention.attention_bias(mem_mask, "masking",
        #                                        dtype=dtype, name='attention_bias2')
        # bias2 = tf.squeeze(bias2, axis=[1, 2])
        input_ta = tf.TensorArray(dtype, time_steps,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(dtype, time_steps,
                                   tensor_array_name="output_array")
        value_ta = tf.TensorArray(dtype, time_steps,
                                  tensor_array_name="value_array")
        alpha_ta = tf.TensorArray(dtype, time_steps,
                                  tensor_array_name="alpha_array")
        coref_ta = tf.TensorArray(dtype, time_steps,
                                  tensor_array_name="coref_array")
        input_ta = input_ta.unstack(inputs)

        if target_copy_hidden_states is None:
            # target_copy_hidden_states = [tf.zeros([batch, 1, time_steps],dtype=tf.float32)]
            target_copy_hidden_states = tf.TensorArray(dtype, time_steps+1, clear_after_read=False,
                                            tensor_array_name="target_copy_hidden_states")
            init_input = tf.zeros([batch,output_size],dtype=tf.float32)
            target_copy_hidden_states = target_copy_hidden_states.write(0, init_input)
            #cache2 = layers.attention.attention(None, init_input, None, output_size,scope='attention2')
        
        def loop_func(t, out_ta, att_ta, cor_ta, val_ta, state, state2, cache_key,target_copy_hidden_states):
            inp_t = input_ta.read(t)
            results = layers.attention.attention(state, memory, bias,
                                                 output_size,
                                                 cache={"key": cache_key})
            # tf.Print(t,['time:', t])
            # results_coref = tf.cond(tf.equal(t,0), lambda: layers.attention.attention(state, init_input, bias2,
            #                                      output_size,
            #                                      cache=None, scope='attention2'),
            #                                     lambda: layers.attention.attention(state, tf.transpose(target_copy_hidden_states.stack(), [1,0,2]), bias2,
            #                                      output_size,
            #                                      cache=None, scope='attention2',reuse=True))
            # pdb.set_trace()
            indices = tf.cond(tf.equal(t,0), lambda: tf.range(1), lambda: tf.range(1,t+1))
            hidden_states=target_copy_hidden_states.gather(indices=indices)
            results_coref = layers.attention.attention(state, tf.transpose(hidden_states, [1,0,2]), None, output_size, cache=None, scope='attention2')
            #pad_tensor=tf.Tensor([[0,0],[0,0],[0, time_steps - t]],shape=[1,3])
            #pdb.set_trace()
            # pad_tensor = tf.zeros([3,2])
            target_copy_attention=results_coref['weight']
            pad_val = tf.cond(tf.equal(t,0), lambda: time_steps-t-1, lambda: time_steps-t)
            pad_tensor = tf.one_hot([-1,1],2, on_value=pad_val)# * (time_steps-t)
            paddings = tf.cast(pad_tensor, tf.int32)
            #pad_tensor = tf.assign(pad_tensor[2,1],time_steps-t)
            #pdb.set_trace()
            target_copy_attention=tf.pad(target_copy_attention, paddings, mode='CONSTANT', constant_values=0)
            
            alpha = results["weight"]
            context = results["value"]
            cell_input = [inp_t, context]
            # pdb.set_trace()
            cell1_input, new_state2 = cell2(cell_input, state2,scope='cell2')
            cell_output, new_state = cell1(cell1_input, state,scope='cell1')
            cell_output = _copy_through(t, sequence_length["target"],
                                        zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state,
                                      new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value,
                                      context)

            # new_target_state = _copy_through(t, sequence_length["target"], target_copy_hidden_states,
            #                           context)
            # target_copy_hidden_states.append(context)
            out_ta = out_ta.write(t, cell_output)
            att_ta = att_ta.write(t, alpha)
            val_ta = val_ta.write(t, new_value)
            cor_ta = cor_ta.write(t, target_copy_attention)
            target_copy_hidden_states = target_copy_hidden_states.write(t+1, context)
            cache_key = tf.identity(cache_key)
            return t + 1, out_ta, att_ta, cor_ta, val_ta, new_state, new_state2, cache_key, target_copy_hidden_states

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, alpha_ta, coref_ta, value_ta, initial_state, initial_state, cache['key'], target_copy_hidden_states)

        outputs = tf.while_loop(lambda t, *_: t < time_steps,loop_func, loop_vars,parallel_iterations=32,swap_memory=True)

        output_final_ta = outputs[1]
        value_final_ta = outputs[4]

        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])

        final_value = value_final_ta.stack()
        final_value.set_shape([None, None, memory.shape[-1].value])
        final_value = tf.transpose(final_value, [1, 0, 2])
        attention_weights = tf.transpose(outputs[2].stack(), [1,0,2])
        coref_weights = tf.transpose(outputs[3].stack(), [1,0,2])
        result = {
            "outputs": final_output,
            "values": final_value,
            "states": [outputs[5], outputs[6]],
            'SrcWeights': attention_weights,
            'CorefWeights': coref_weights
        }

    return result


def _step(cell1, cell2, inputs, memory, sequence_length, state1, state2, dtype=None,
             scope=None, target_copy_hidden_states=None):
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell2.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)
    target_copy_attentions = []
    # pdb.set_trace()
    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])
        mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(memory)[1],
                                    dtype=dtype)
        bias = layers.attention.attention_bias(mem_mask, "masking",
                                               dtype=dtype)
        bias = tf.squeeze(bias, axis=[1, 2])
        cache = layers.attention.attention(None, memory, None, output_size)

        # bias2 = layers.attention.attention_bias(mem_mask, "masking",
        #                                        dtype=dtype, name='attention_bias2')
        # bias2 = tf.squeeze(bias2, axis=[1, 2])
        input_ta = tf.TensorArray(dtype, time_steps,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(dtype, time_steps,
                                   tensor_array_name="output_array")
        value_ta = tf.TensorArray(dtype, time_steps,
                                  tensor_array_name="value_array")
        alpha_ta = tf.TensorArray(dtype, time_steps,
                                  tensor_array_name="alpha_array")
        coref_ta = tf.TensorArray(dtype, time_steps,
                                  tensor_array_name="coref_array")
        input_ta = input_ta.unstack(inputs)

        if target_copy_hidden_states is None:
            # target_copy_hidden_states = [tf.zeros([batch, 1, time_steps],dtype=tf.float32)]
            target_copy_hidden_states = tf.TensorArray(dtype, time_steps+1, clear_after_read=False,
                                            tensor_array_name="target_copy_hidden_states")
            init_input = tf.zeros([batch,output_size],dtype=tf.float32)
            target_copy_hidden_states = target_copy_hidden_states.write(0, init_input)
            #cache2 = layers.attention.attention(None, init_input, None, output_size,scope='attention2')
        
        def loop_func(t, out_ta, att_ta, cor_ta, val_ta, state, state2, cache_key,target_copy_hidden_states):
            inp_t = input_ta.read(t)
            results = layers.attention.attention(state, memory, bias,
                                                 output_size,
                                                 cache={"key": cache_key})
            # tf.Print(t,['time:', t])
            # results_coref = tf.cond(tf.equal(t,0), lambda: layers.attention.attention(state, init_input, bias2,
            #                                      output_size,
            #                                      cache=None, scope='attention2'),
            #                                     lambda: layers.attention.attention(state, tf.transpose(target_copy_hidden_states.stack(), [1,0,2]), bias2,
            #                                      output_size,
            #                                      cache=None, scope='attention2',reuse=True))
            # pdb.set_trace()
            indices = tf.cond(tf.equal(t,0), lambda: tf.range(1), lambda: tf.range(1,t+1))
            hidden_states=target_copy_hidden_states.gather(indices=indices)
            results_coref = layers.attention.attention(state, tf.transpose(hidden_states, [1,0,2]), None, output_size, cache=None, scope='attention2')
            #pad_tensor=tf.Tensor([[0,0],[0,0],[0, time_steps - t]],shape=[1,3])
            #pdb.set_trace()
            # pad_tensor = tf.zeros([3,2])
            target_copy_attention=results_coref['weight']
            pad_val = tf.cond(tf.equal(t,0), lambda: time_steps-t-1, lambda: time_steps-t)
            pad_tensor = tf.one_hot([-1,1],2, on_value=pad_val)# * (time_steps-t)
            paddings = tf.cast(pad_tensor, tf.int32)
            #pad_tensor = tf.assign(pad_tensor[2,1],time_steps-t)
            #pdb.set_trace()
            target_copy_attention=tf.pad(target_copy_attention, paddings, mode='CONSTANT', constant_values=0)
            
            alpha = results["weight"]
            context = results["value"]
            cell_input = [inp_t, context]
            # pdb.set_trace()
            cell1_input, new_state2 = cell2(cell_input, state2,scope='cell2')
            cell_output, new_state = cell1(cell1_input, state,scope='cell1')
            cell_output = _copy_through(t, sequence_length["target"],
                                        zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state,
                                      new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value,
                                      context)

            # new_target_state = _copy_through(t, sequence_length["target"], target_copy_hidden_states,
            #                           context)
            # target_copy_hidden_states.append(context)
            out_ta = out_ta.write(t, cell_output)
            att_ta = att_ta.write(t, alpha)
            val_ta = val_ta.write(t, new_value)
            cor_ta = cor_ta.write(t, target_copy_attention)
            target_copy_hidden_states = target_copy_hidden_states.write(t+1, context)
            cache_key = tf.identity(cache_key)
            return t + 1, out_ta, att_ta, cor_ta, val_ta, new_state, new_state2, cache_key, target_copy_hidden_states

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, alpha_ta, coref_ta, value_ta, state1, state2, cache['key'], target_copy_hidden_states)

        outputs = tf.while_loop(lambda t, *_: t < time_steps,loop_func, loop_vars,parallel_iterations=32,swap_memory=True)

        output_final_ta = outputs[1]
        value_final_ta = outputs[4]

        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])

        final_value = value_final_ta.stack()
        final_value.set_shape([None, None, memory.shape[-1].value])
        final_value = tf.transpose(final_value, [1, 0, 2])
        attention_weights = tf.transpose(outputs[2].stack(), [1,0,2])
        coref_weights = tf.transpose(outputs[3].stack(), [1,0,2])
        result = {
            "outputs": final_output,
            "values": final_value,
            "states": [outputs[5], outputs[6]],
            'SrcWeights': attention_weights,
            'CorefWeights': coref_weights
        }

    return result


# def _decoder(cell, inputs, memory, sequence_length, initial_state, dtype=None,
#              scope=None, target_copy_hidden_states=None):
#     # Assume that the underlying cell is GRUCell-like
#     batch = tf.shape(inputs)[0]
#     time_steps = tf.shape(inputs)[1]
#     dtype = dtype or inputs.dtype
#     output_size = cell.output_size
#     zero_output = tf.zeros([batch, output_size], dtype)
#     zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)
#     target_copy_attentions = []
#     # pdb.set_trace()
#     with tf.variable_scope(scope or "decoder", dtype=dtype):
#         inputs = tf.transpose(inputs, [1, 0, 2])
#         mem_mask = tf.sequence_mask(sequence_length["source"],
#                                     maxlen=tf.shape(memory)[1],
#                                     dtype=dtype)
#         bias = layers.attention.attention_bias(mem_mask, "masking",
#                                                dtype=dtype)
#         bias = tf.squeeze(bias, axis=[1, 2])
#         cache = layers.attention.attention(None, memory, None, output_size)

#         # bias2 = layers.attention.attention_bias(mem_mask, "masking",
#         #                                        dtype=dtype, name='attention_bias2')
#         # bias2 = tf.squeeze(bias2, axis=[1, 2])
        

#         input_ta = tf.TensorArray(dtype, time_steps,
#                                   tensor_array_name="input_array")
#         output_ta = tf.TensorArray(dtype, time_steps,
#                                    tensor_array_name="output_array")
#         value_ta = tf.TensorArray(dtype, time_steps,
#                                   tensor_array_name="value_array")
#         alpha_ta = tf.TensorArray(dtype, time_steps,
#                                   tensor_array_name="alpha_array")
#         coref_ta = tf.TensorArray(dtype, time_steps,
#                                   tensor_array_name="coref_array")
#         input_ta = input_ta.unstack(inputs)
#         initial_state = layers.nn.linear(initial_state, output_size, True,
#                                          False, scope="s_transform")
#         initial_state = tf.tanh(initial_state)
#         if target_copy_hidden_states is None:
#             # target_copy_hidden_states = [tf.zeros([batch, 1, time_steps],dtype=tf.float32)]
#             target_copy_hidden_states = tf.TensorArray(dtype, time_steps+1, clear_after_read=False,
#                                             tensor_array_name="target_copy_hidden_states")
#             init_input = tf.zeros([batch,output_size],dtype=tf.float32)
#             target_copy_hidden_states = target_copy_hidden_states.write(0, init_input)
#             #cache2 = layers.attention.attention(None, init_input, None, output_size,scope='attention2')
        
#         def loop_func(t, out_ta, att_ta, cor_ta, val_ta, state, cache_key,target_copy_hidden_states):
#             inp_t = input_ta.read(t)
#             results = layers.attention.attention(state, memory, bias,
#                                                  output_size,
#                                                  cache={"key": cache_key})
#             # tf.Print(t,['time:', t])
#             # results_coref = tf.cond(tf.equal(t,0), lambda: layers.attention.attention(state, init_input, bias2,
#             #                                      output_size,
#             #                                      cache=None, scope='attention2'),
#             #                                     lambda: layers.attention.attention(state, tf.transpose(target_copy_hidden_states.stack(), [1,0,2]), bias2,
#             #                                      output_size,
#             #                                      cache=None, scope='attention2',reuse=True))
#             # pdb.set_trace()
#             indices = tf.cond(tf.equal(t,0), lambda: tf.range(1), lambda: tf.range(1,t+1))
#             hidden_states=target_copy_hidden_states.gather(indices=indices)
#             results_coref = layers.attention.attention(state, tf.transpose(hidden_states, [1,0,2]), None, output_size, cache=None, scope='attention2')
#             #pad_tensor=tf.Tensor([[0,0],[0,0],[0, time_steps - t]],shape=[1,3])
#             #pdb.set_trace()
#             # pad_tensor = tf.zeros([3,2])
#             target_copy_attention=results_coref['weight']
#             pad_val = tf.cond(tf.equal(t,0), lambda: time_steps-t-1, lambda: time_steps-t)
#             pad_tensor = tf.one_hot([-1,1],2, on_value=pad_val)# * (time_steps-t)
#             paddings = tf.cast(pad_tensor, tf.int32)
#             #pad_tensor = tf.assign(pad_tensor[2,1],time_steps-t)
#             #pdb.set_trace()
#             target_copy_attention=tf.pad(target_copy_attention, paddings, mode='CONSTANT', constant_values=0)
            
#             alpha = results["weight"]
#             context = results["value"]
#             cell_input = [inp_t, context]
#             # pdb.set_trace()
#             cell_output, new_state = cell(cell_input, state)
#             cell_output = _copy_through(t, sequence_length["target"],
#                                         zero_output, cell_output)
#             new_state = _copy_through(t, sequence_length["target"], state,
#                                       new_state)
#             new_value = _copy_through(t, sequence_length["target"], zero_value,
#                                       context)

#             # new_target_state = _copy_through(t, sequence_length["target"], target_copy_hidden_states,
#             #                           context)
#             # target_copy_hidden_states.append(context)
#             out_ta = out_ta.write(t, cell_output)
#             att_ta = att_ta.write(t, alpha)
#             val_ta = val_ta.write(t, new_value)
#             cor_ta = cor_ta.write(t, target_copy_attention)
#             target_copy_hidden_states = target_copy_hidden_states.write(t+1, context)
#             cache_key = tf.identity(cache_key)
#             return t + 1, out_ta, att_ta, cor_ta, val_ta, new_state, cache_key, target_copy_hidden_states

#         time = tf.constant(0, dtype=tf.int32, name="time")
#         loop_vars = (time, output_ta, alpha_ta, coref_ta, value_ta, initial_state, cache['key'], target_copy_hidden_states)

#         outputs = tf.while_loop(lambda t, *_: t < time_steps,loop_func, loop_vars,parallel_iterations=32,swap_memory=True)

#         output_final_ta = outputs[1]
#         value_final_ta = outputs[4]

#         final_output = output_final_ta.stack()
#         final_output.set_shape([None, None, output_size])
#         final_output = tf.transpose(final_output, [1, 0, 2])

#         final_value = value_final_ta.stack()
#         final_value.set_shape([None, None, memory.shape[-1].value])
#         final_value = tf.transpose(final_value, [1, 0, 2])
#         attention_weights = tf.transpose(outputs[2].stack(), [1,0,2])
#         coref_weights = tf.transpose(outputs[3].stack(), [1,0,2])
#         result = {
#             "outputs": final_output,
#             "values": final_value,
#             "initial_state": initial_state,
#             'SrcWeights': attention_weights,
#             'CorefWeights': coref_weights
#         }

#     return result


def model_graph(features, mode, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])
    dtype = tf.get_variable_scope().dtype

    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding",
                                  [src_vocab_size, params.embedding_size])
        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

    with tf.variable_scope("target_embedding"):
        tgt_emb = tf.get_variable("embedding",
                                  [tgt_vocab_size, params.embedding_size])
        tgt_bias = tf.get_variable("bias", [params.embedding_size])
        tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])

    src_inputs = tf.nn.bias_add(src_inputs, src_bias)
    tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)

    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
        tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

    # encoder
    cell_fw = layers.rnn_cell.LegacyGRUCell(params.hidden_size)
    cell_bw = layers.rnn_cell.LegacyGRUCell(params.hidden_size)

    if params.use_variational_dropout:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=dtype
        )
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=dtype
        )

    encoder_output = _encoder(cell_fw, cell_bw, src_inputs,
                              features["source_length"], dtype=dtype)

    # decoder
    cell = layers.rnn_cell.LegacyGRUCell(params.hidden_size)

    if params.use_variational_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            # input + context
            input_size=params.embedding_size + 2 * params.hidden_size,
            dtype=dtype
        )

    length = {
        "source": features["source_length"],
        "target": features["target_length"]
    }
    initial_state = encoder_output["final_states"]["backward"]
    decoder_output = _decoder(cell, tgt_inputs, encoder_output["annotation"],
                              length, initial_state, dtype=dtype)

    # Shift left
    shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
    shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]

    all_outputs = tf.concat(
        [
            tf.expand_dims(decoder_output["initial_state"], axis=1),
            decoder_output["outputs"],
        ],
        axis=1
    )
    shifted_outputs = all_outputs[:, :-1, :]

    maxout_features = [
        shifted_tgt_inputs,
        shifted_outputs,
        decoder_output["values"]
    ]
    maxout_size = params.hidden_size // params.maxnum

    if mode == "infer":
        # Special case for non-incremental decoding
        maxout_features = [
            shifted_tgt_inputs[:, -1, :],
            shifted_outputs[:, -1, :],
            decoder_output["values"][:, -1, :]
        ]
        maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                                  concat=False)
        readout = layers.nn.linear(maxhid, params.embedding_size, False,
                                   False, scope="deepout")

        # Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True, False,
                                  scope="softmax")

        return tf.nn.log_softmax(logits)

    maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                              concat=False)
    readout = layers.nn.linear(maxhid, params.embedding_size, False, False,
                               scope="deepout")

    if params.dropout and not params.use_variational_dropout:
        readout = tf.nn.dropout(readout, 1.0 - params.dropout)

    # Prediction
    logits = layers.nn.linear(readout, tgt_vocab_size, True, False,
                              scope="softmax")
    logits = tf.reshape(logits, [-1, tgt_vocab_size])
    labels = features["target"]

    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(labels))
    tgt_mask = tf.to_float(
        tf.sequence_mask(
            features["target_length"],
            maxlen=tf.shape(features["target"])[1]
        )
    )

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss

