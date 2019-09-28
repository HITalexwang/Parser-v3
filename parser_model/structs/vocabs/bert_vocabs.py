#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import os
import codecs
import zipfile
import gzip
import re

from parser_model.structs.buckets import ListMultibucket

try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl
from collections import Counter

import numpy as np
import tensorflow as tf
import h5py
 
from parser_model.structs.vocabs.base_vocabs import CountVocab
from . import conllu_vocabs as cv
from parser_model.neural import embeddings


import bert
from bert import modeling
from bert import tokenization

import tensorflow as tf
# import tensorflow_hub as hub


#***************************************************************
class BERTVocab(CountVocab):
  """"""

  _save_str = 'BERT'

  #=============================================================
  def __init__(self, config=None):
    """"""

    super(BERTVocab, self).__init__(config=config)

    self._wordpiece_placeholder = tf.placeholder(tf.int32, [None, None], self.classname + '_wordpiece')
    self._first_index_placeholder = tf.placeholder(tf.int32, [None, None], self.classname + '_first_index')
    self._full_tokenizer = bert.tokenization.FullTokenizer(vocab_file=self.vocab_file,
                                                           do_lower_case=self.do_lower_case)
    self._bert_module = None
    self._bert_config = modeling.BertConfig.from_json_file(self.config_file)
    # with tf.variable_scope(self._config._defaults['network_class']): # hack in the top module's scope
    #   self._bert_module = hub.Module(self.bert_hub_path, trainable=self.trainable, name='BERT_module')
    #   tokenization_info = self._bert_module(signature="tokenization_info", as_dict=True)
    #   config = tf.ConfigProto()
    #   config.gpu_options.allow_growth = True
    #   with tf.Session(config=config) as sess:
    #     self.vocab_file, self.do_lower_case = sess.run([tokenization_info["vocab_file"],
    #                                                     tokenization_info["do_lower_case"]])

    return

  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""

    with tf.variable_scope(variable_scope or self.classname, reuse=tf.AUTO_REUSE if reuse else False):
      self._bert_module = modeling.BertModel(config=self._bert_config,
                                             is_training=self.trainable,
                                             input_ids=self._wordpiece_placeholder,
                                             input_mask=tf.cast(self._wordpiece_placeholder > 0, tf.int32),
                                             token_type_ids=tf.zeros_like(self._wordpiece_placeholder),
                                             use_one_hot_embeddings=None,
                                             scope='bert')
    # bert_inputs = dict(
    #   input_ids=self._wordpiece_placeholder,
    #   input_mask=tf.cast(self._wordpiece_placeholder > 0, tf.int32),
    #   segment_ids=tf.zeros_like(self._wordpiece_placeholder)
    # )
    # bert_outputs = self._bert_module(
    #   inputs=bert_inputs,
    #   signature="tokens",
    #   as_dict=True
    # )
    # layer = bert_outputs['sequence_output']

    layer = self._bert_module.get_sequence_output()
    layer = tf.batch_gather(layer, self._first_index_placeholder)

    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    if embed_keep_prob < 1:
      layer = self.drop_func(layer, embed_keep_prob)
    return layer

  #=============================================================
  def count(self, train_conllus):
    """"""
    for train_conllu in train_conllus:
      with codecs.open(train_conllu, encoding='utf-8', errors='ignore') as f:
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            line = line.split('\t')
            token = line[self.conllu_idx] # conllu_idx is provided by the CoNLLUVocab
            self._count(token)
    self.index_by_counts()
    return True

  def _count(self, token):
    if not self.cased:
      token = token.lower()
    self.counts[token] += 1
    return

  #=============================================================
  def load(self):
    """"""

    if super(BERTVocab, self).load():
      self._loaded = True
      return True
    else:
      if os.path.exists(self.token_vocab_savename):
        token_vocab_filename = self.token_vocab_savename
      elif self.token_vocab_loadname and os.path.exists(self.token_vocab_loadname):
        token_vocab_filename = self.token_vocab_loadname
      else:
        self._loaded = False
        return False

      with codecs.open(token_vocab_filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
          line = line.rstrip()
          if line:
            match = re.match('(.*)\s([0-9]*)', line)
            token = match.group(1)
            count = int(match.group(2))
            self._count(token)
            self._count(token.upper())
      self.index_by_counts(dump=True)
      self._loaded = True
      return True

  #=============================================================
  def add(self, token):
    """"""

    # wordpieces = self._full_tokenizer.wordpiece_tokenizer.tokenize(token)
    # wordpiece_indices = self._full_tokenizer.convert_tokens_to_ids(wordpieces)
    # token_index = self._multibucket.add(wordpiece_indices, wordpieces)
    if token in self._str2idx:
      return self._str2idx[token]
    token_index = len(self._str2idx)
    self._str2idx[token] = token_index
    self._idx2str[token_index] = token
    return token_index

  #=============================================================
  def token(self, index):
    """"""

    return self._idx2str[index]

  #=============================================================
  def index(self, token):
    """"""

    return self._str2idx[token]

  #=============================================================
  def set_placeholders(self, indices, tokens, feed_dict={}):
    """"""

    # unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
    # feed_dict[self.placeholder] = inverse_indices.reshape(indices.shape)
    # self._multibucket.set_placeholders(unique_indices, feed_dict=feed_dict)
    # self._multibucket.set_placeholders(indices, feed_dict=feed_dict)

    all_wordpiece_list = []
    all_first_index_list = []
    for i, data in enumerate(indices):
      wordpiece_list = ['[CLS]']
      first_index_list = []
      for j, word in enumerate(data):
        if word == 0:
          break
        if word == self.PAD_IDX:
          token = '[PAD]'
        elif word == self.UNK_IDX:
          # token = '[UNK]'
          token = tokens[i][j]
        elif word == self.ROOT_IDX:
          token = '[unused1]'
        else:
          token = self.token(word)
        first_index_list.append(len(wordpiece_list))
        wordpiece_list += self._full_tokenizer.wordpiece_tokenizer.tokenize(token)
      wordpiece_list += ['[SEP]']
      wordpiece_list = self._full_tokenizer.convert_tokens_to_ids(wordpiece_list)
      all_wordpiece_list.append(wordpiece_list)
      all_first_index_list.append(first_index_list)

    all_wordpiece_max_len = max([len(a) for a in all_wordpiece_list])
    all_wordpiece = np.stack(
      [np.pad(a, (0, all_wordpiece_max_len - len(a)), 'constant', constant_values=0) for a in all_wordpiece_list])
    feed_dict[self._wordpiece_placeholder] = all_wordpiece

    # all_first_index_max_len = max([len(a) for a in all_first_index_list])
    all_first_index_max_len = indices.shape[-1]
    all_first_index = np.stack(
      [np.pad(a, (0, all_first_index_max_len - len(a)), 'constant', constant_values=0) for a in all_first_index_list])
    feed_dict[self._first_index_placeholder] = all_first_index

    feed_dict[self.placeholder] = indices

    return feed_dict

  #=============================================================
  # def open(self):
  #   """"""
  #
  #   # self._multibucket.open()
  #   return self

  #=============================================================
  # def close(self):
  #   """"""
  #
  #   # self._multibucket.close()
  #   return

  #=============================================================
  @property
  def token_vocab_savename(self):
    return os.path.join(self.save_dir, self.field+'-tokens.lst')
  @property
  def token_vocab_loadname(self):
    return self._config.getstr(self, 'token_vocab_loadname')
  # @property
  # def max_buckets(self):
  #   return self._config.getint(self, 'max_buckets')
  @property
  def embed_keep_prob(self):
    return self._config.getfloat(self, 'embed_keep_prob')
  # @property
  # def conv_keep_prob(self):
  #   return self._config.getfloat(self, 'conv_keep_prob')
  # @property
  # def recur_keep_prob(self):
  #   return self._config.getfloat(self, 'recur_keep_prob')
  # @property
  # def linear_keep_prob(self):
  #   return self._config.getfloat(self, 'linear_keep_prob')
  # @property
  # def output_keep_prob(self):
  #   return self._config.getfloat(self, 'output_keep_prob')
  # @property
  # def n_layers(self):
  #   return self._config.getint(self, 'n_layers')
  # @property
  # def first_layer_conv_width(self):
  #   return self._config.getint(self, 'first_layer_conv_width')
  # @property
  # def conv_width(self):
  #   return self._config.getint(self, 'conv_width')
  # @property
  # def embed_size(self):
  #   return self._config.getint(self, 'embed_size')
  # @property
  # def recur_size(self):
  #   return self._config.getint(self, 'recur_size')
  # @property
  # def output_size(self):
  #   return self._config.getint(self, 'output_size')
  # @property
  # def hidden_size(self):
  #   return self._config.getint(self, 'hidden_size')
  # @property
  # def bidirectional(self):
  #   return self._config.getboolean(self, 'bidirectional')
  @property
  def drop_func(self):
    drop_func = self._config.getstr(self, 'drop_func')
    if hasattr(embeddings, drop_func):
      return getattr(embeddings, drop_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(embeddings.__name__, drop_func))
  # @property
  # def recur_func(self):
  #   recur_func = self._config.getstr(self, 'recur_func')
  #   if hasattr(nonlin, recur_func):
  #     return getattr(nonlin, recur_func)
  #   else:
  #     raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, recur_func))
  # @property
  # def highway_func(self):
  #   highway_func = self._config.getstr(self, 'highway_func')
  #   if hasattr(nonlin, highway_func):
  #     return getattr(nonlin, highway_func)
  #   else:
  #     raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, highway_func))
  # @property
  # def output_func(self):
  #   output_func = self._config.getstr(self, 'output_func')
  #   if hasattr(nonlin, output_func):
  #     return getattr(nonlin, output_func)
  #   else:
  #     raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, output_func))
  # @property
  # def recur_cell(self):
  #   recur_cell = self._config.getstr(self, 'recur_cell')
  #   if hasattr(recurrent, recur_cell):
  #     return getattr(recurrent, recur_cell)
  #   else:
  #     raise AttributeError("module '{}' has no attribute '{}'".format(recurrent.__name__, recur_func))
  # @property
  # def drop_type(self):
  #   return self._config.getstr(self, 'drop_type')
  # @property
  # def bilin(self):
  #   return self._config.getboolean(self, 'bilin')
  # @property
  # def cifg(self):
  #   return self._config.getboolean(self, 'cifg')
  # @property
  # def highway(self):
  #   return self._config.getboolean(self, 'highway')
  # @property
  # def squeeze_type(self):
  #   return self._config.getstr(self, 'squeeze_type')
  @property
  def bert_hub_path(self):
    return self._config.getstr(self, 'bert_hub_path')
  @property
  def trainable(self):
    return self._config.getboolean(self, 'trainable')
  @property
  def vocab_file(self):
    return self._config.getstr(self, 'vocab_file')
  @property
  def do_lower_case(self):
    return self._config.getboolean(self, 'do_lower_case')
  @property
  def config_file(self):
    return self._config.getstr(self, 'config_file')
  @property
  def pretrained_ckpt(self):
    return self._config.getstr(self, 'pretrained_ckpt')

# #***************************************************************
# class GraphBERTVocab(BERTVocab):
#   """"""
#
#   def _collect_tokens(self, node):
#     node = node.split('|')
#     for edge in node:
#       edge = edge.split(':', 1)
#       head, rel = edge
#       self.counts.update(rel)
#
# #***************************************************************
class FormBERTVocab(BERTVocab, cv.FormVocab):
  pass
class LemmaBERTVocab(BERTVocab, cv.LemmaVocab):
  pass
class UPOSBERTVocab(BERTVocab, cv.UPOSVocab):
  pass
class XPOSBERTVocab(BERTVocab, cv.XPOSVocab):
  pass
class DeprelBERTVocab(BERTVocab, cv.DeprelVocab):
  pass
