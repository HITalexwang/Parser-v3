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

from parser_model.base_network import BaseNetwork
from parser_model.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser_model.neural import graph_transformer

#***************************************************************
class GraphParserNetwork(BaseNetwork):
  """"""
  
  #=============================================================
  def build_graph(self, input_network_outputs={}, reuse=True):
    """"""
    
    with tf.variable_scope('Embeddings'):

      #pos-tag embedding + word embedding
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

      #word embedding
      else:
        input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]


      for input_network, output in input_network_outputs:
        with tf.variable_scope(input_network.classname):
          input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
      #print ([t.shape.as_list() for t in input_tensors])
      layer = tf.concat(input_tensors, 2)

    # pr_1 = tf.print('\n=======\n', 'layer\n', tf.shape(layer), '\n=======\n')
    # pr_2 = tf.print('\n=======\n', 'self.id_vocab.placeholder\n', tf.shape(self.id_vocab.placeholder), '\n=======\n')
    # pr_3 = tf.print('\n=======\n', 'self._input_vocabs[0].placeholder\n', tf.shape(self._input_vocabs[0].placeholder), '\n=======\n')
    # pr_4 = tf.print('\n=======\n', 'self._input_vocabs[0]._wordpiece_placeholder\n', tf.shape(self._input_vocabs[0]._wordpiece_placeholder),
    #                 '\n=======\n')
    # pr_5 = tf.print('\n=======\n', 'self._input_vocabs[0]._first_index_placeholder\n', tf.shape(self._input_vocabs[0]._first_index_placeholder),
    #                 '\n=======\n')
    #
    # with tf.control_dependencies([pr_1, pr_2, pr_3, pr_4, pr_5]):

    n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keep_dims=True))
    batch_size, bucket_size, input_size = nn.get_sizes(layer)
    layer *= input_size / (n_nonzero + tf.constant(1e-12))
    
    token_weights = nn.greater(self.id_vocab.placeholder, 0)
    tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
    n_tokens = tf.reduce_sum(tokens_per_sequence)
    n_sequences = tf.count_nonzero(tokens_per_sequence)
    seq_lengths = tokens_per_sequence+1

    # this works as the mask (shape = [batch_size, bucket_size])
    root_weights = token_weights + (1-nn.greater(tf.range(bucket_size), 0))
    # this is mask for arc/label prediction (shape = [batch_size, bucket_size, bucket_size])
    token_weights3D = tf.expand_dims(token_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
    tokens = {'n_tokens': n_tokens,
              'tokens_per_sequence': tokens_per_sequence,
              'token_weights': token_weights,
              'token_weights3D': token_weights,
              'n_sequences': n_sequences}
    
    # conv_keep_prob = 1. if reuse else self.conv_keep_prob
    # recur_keep_prob = 1. if reuse else self.recur_keep_prob
    # recur_include_prob = 1. if reuse else self.recur_include_prob
    #
    # for i in six.moves.range(self.n_layers):
    #   conv_width = self.first_layer_conv_width if not i else self.conv_width
    #   with tf.variable_scope('RNN-{}'.format(i)):
    #     layer, _ = recurrent.directed_RNN(layer, self.recur_size, seq_lengths,
    #                                       bidirectional=self.bidirectional,
    #                                       recur_cell=self.recur_cell,
    #                                       conv_width=conv_width,
    #                                       recur_func=self.recur_func,
    #                                       conv_keep_prob=conv_keep_prob,
    #                                       recur_include_prob=recur_include_prob,
    #                                       recur_keep_prob=recur_keep_prob,
    #                                       cifg=self.cifg,
    #                                       highway=self.highway,
    #                                       highway_func=self.highway_func,
    #                                       bilin=self.bilin)
    
    config = graph_transformer.GraphTransformerConfig(hidden_size=self.hidden_size,
                                                      num_hidden_layers=self.n_layers,
                                                      num_attention_heads=self.n_attention_heads,
                                                      intermediate_size=self.intermediate_size,
                                                      hidden_act="gelu",
                                                      hidden_dropout_prob=self.hidden_dropout_prob,
                                                      attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                                                      acc_mask_dropout_prob=self.acc_mask_dropout_prob,
                                                      max_position_embeddings=self.max_position_embeddings,
                                                      initializer_range=0.02,
                                                      supervision=self.supervision,
                                                      smoothing_rate=self.smoothing_rate,
                                                      acc_inters=self.acc_inters)

    output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
    outputs = {}

    with tf.variable_scope('Transformer'):
      # shape = [batch_size, seq_len, seq_len]
      input_mask_3D = tf.expand_dims(root_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
      if self.supervision.startswith('graph'):
        acc_matrices = output_fields['semhead'].placeholder
      else:
        acc_matrices = output_fields['semhead'].accessible_placeholders
      transformer = graph_transformer.GraphTransformer(config, not reuse, layer, 
                                                        input_mask=input_mask_3D,
                                                        accessible_matrices=acc_matrices)
      # shape = [batch_size, seq_len, hidden_size]
      layer = transformer.get_sequence_output()
      acc_outputs = None
      if self.supervision != 'none':
        acc_outputs = transformer.get_accessible_outputs()
        if self.supervision.startswith('graph'):
          acc_outputs['acc_loss'] = tf.reduce_sum(acc_outputs['acc_loss'])
        else:
          for field in acc_outputs:
            acc_outputs[field] = tf.reduce_sum(acc_outputs[field])
    
    with tf.variable_scope('Classifiers'):
      if 'semrel' in output_fields:
        vocab = output_fields['semrel']
        if vocab.factorized:
          head_vocab = output_fields['semhead']
          with tf.variable_scope('Unlabeled'):
            unlabeled_outputs = head_vocab.get_bilinear_discriminator(
              layer,
              token_weights=token_weights3D,
              reuse=reuse)
          with tf.variable_scope('Labeled'):
            labeled_outputs = vocab.get_bilinear_classifier(
              layer, unlabeled_outputs,
              token_weights=token_weights3D,
              reuse=reuse)
        else:
          labeled_outputs = vocab.get_unfactored_bilinear_classifier(layer, head_vocab.placeholder,
            token_weights=token_weights3D,
            reuse=reuse)
        outputs['semgraph'] = labeled_outputs
        if acc_outputs is not None:
          outputs['semgraph']['loss'] += acc_outputs['acc_loss']
          for field in acc_outputs:
            outputs['semgraph'][field] = acc_outputs[field]
        self._evals.add('semgraph')
      elif 'semhead' in output_fields:
        vocab = output_fields['semhead']
        outputs[vocab.classname] = vocab.get_bilinear_classifier(
          layer,
          token_weights=token_weights3D,
          reuse=reuse)
        self._evals.add('semhead')
    
    return outputs, tokens
  
  #=============================================================
  @property
  def sum_pos(self):
    return self._config.getboolean(self, 'sum_pos')
  @property
  def hidden_size(self):
    return self._config.getint(self, 'hidden_size')
  @property
  def intermediate_size(self):
    return self._config.getint(self, 'intermediate_size')
  @property
  def n_layers(self):
    return self._config.getint(self, 'n_layers')
  @property
  def n_attention_heads(self):
    return self._config.getint(self, 'n_attention_heads')
  @property
  def hidden_dropout_prob(self):
    return self._config.getfloat(self, 'hidden_dropout_prob')
  @property
  def attention_probs_dropout_prob(self):
    return self._config.getfloat(self, 'attention_probs_dropout_prob')
  @property
  def acc_mask_dropout_prob(self):
    return self._config.getfloat(self, 'acc_mask_dropout_prob')
  @property
  def max_position_embeddings(self):
    return self._config.getint(self, 'max_position_embeddings')
  @property
  def supervision(self):
    return self._config.getstr(self, 'supervision')
  @property
  def smoothing_rate(self):
    return self._config.getfloat(self, 'smoothing_rate')
  @property
  def acc_inters(self):
    return [float(f) for f in self._config.getlist(self, 'acc_inters')]