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
import tensorflow_hub as hub

from parser_model.base_network import BaseNetwork
from parser_model.neural import nn, nonlin, embeddings, recurrent, classifiers

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
#***************************************************************
class BERTNetwork(BaseNetwork):
  """"""
  
  _evals = set()
  
  #=============================================================
  def build_graph(self, input_network_outputs={}, reuse=True):
    """"""
    with tf.variable_scope('Embeddings'):
        input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]

        for input_network, output in input_network_outputs:
            with tf.variable_scope(input_network.classname):
                input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
        layer = tf.concat(input_tensors, 2)

    n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keep_dims=True))
    batch_size, bucket_size, input_size = nn.get_sizes(layer)
    layer *= input_size / (n_nonzero + tf.constant(1e-12))

    token_weights = nn.greater(self.id_vocab.placeholder, 0)
    tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
    n_tokens = tf.reduce_sum(tokens_per_sequence)
    n_sequences = tf.count_nonzero(tokens_per_sequence)
    seq_lengths = tokens_per_sequence + 1

    root_weights = token_weights + (1 - nn.greater(tf.range(bucket_size), 0))
    token_weights3D = tf.expand_dims(token_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
    tokens = {'n_tokens': n_tokens,
              'tokens_per_sequence': tokens_per_sequence,
              'token_weights': token_weights,
              'token_weights3D': token_weights,
              'n_sequences': n_sequences}

    conv_keep_prob = 1. if reuse else self.conv_keep_prob
    recur_keep_prob = 1. if reuse else self.recur_keep_prob
    recur_include_prob = 1. if reuse else self.recur_include_prob

    
    output_vocabs = {vocab.field: vocab for vocab in self.output_vocabs}
    outputs = {}
    # with tf.variable_scope('Classifiers'):
    #   if 'FormTokenVocab' in output_vocabs:
    #     vocab = output_vocabs['FormTokenVocab']
    #     outputs[vocab.field] = vocab.get_sampled_linear_classifier(
    #       layer, self.n_samples,
    #       token_weights=token_weights,
    #       reuse=reuse)
    #     self._evals.add('form')
    print(tokens,'\n',outputs)

    return outputs, tokens
  
  #=============================================================
  @property
  def n_samples(self):
    return self._config.getint(self, 'n_samples')
