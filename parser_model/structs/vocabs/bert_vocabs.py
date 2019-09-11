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

import os
import codecs
import zipfile
import gzip
import re
try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl
from collections import Counter

import numpy as np
import tensorflow as tf
import h5py
 
from parser_model.structs.vocabs.base_vocabs import SetVocab
from . import conllu_vocabs as cv
from parser_model.neural import embeddings


import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

import tensorflow as tf
import tensorflow_hub as hub

BERT_MODEL_HUB = "/Users/longxud/Documents/Code/bert/hub_bert_cased_L-12_H-768_A-12"

#***************************************************************
class BERTVocab(SetVocab):
  """"""
  #=============================================================
  def __init__(self, config=None):
    """"""

    self._elmo_test_filename = config.getstr(self, 'elmo_test_filename')
    #print ('Elmo test:',self._elmo_test_filename)
    if self._elmo_test_filename:
      self._conllu_files = config.getlist(self, 'conllu_files')
      #print (self._conllu_files)
    else:
      self._elmo_train_filename = config.getstr(self, 'elmo_train_filename')
      self._elmo_dev_filename = config.getstr(self, 'elmo_dev_filename')
      self._train_conllus = config.getlist(self, 'train_conllus')
      self._dev_conllus = config.getlist(self, 'dev_conllus')
      #print (self._train_conllus, self._dev_conllus)

    super(BERTVocab, self).__init__(config=config)
    self._name = config.getstr(self, 'name')
    self.variable = None
    return
  
  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""
    
    # Default override
    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    input_embed_keep_prob = self.input_embed_keep_prob
    with tf.variable_scope(variable_scope or self.field):
      if self.variable is None:
        with tf.device('/cpu:0'):
          self.variable = tf.Variable(self.embed_placeholder, name=self.name+'Elmo', trainable=False)
          tf.add_to_collection('non_save_variables', self.variable)
      layer = embeddings.pretrained_embedding_lookup(self.variable, self.linear_size,
                                                     self.placeholder,
                                                     name=self.name,
                                                     reuse=reuse,
                                                     input_embed_keep_prob=input_embed_keep_prob)
      if embed_keep_prob < 1:
        layer = self.drop_func(layer, embed_keep_prob)
    return layer

  #=============================================================
  def load(self):
    self._loaded = False
    return False

  #=============================================================
  @property
  def elmo_train_filename(self):
    return self._config.getstr(self, 'elmo_train_filename')
  @property
  def elmo_dev_filename(self):
    return self._config.getstr(self, 'elmo_dev_filename')
  @property
  def elmo_test_filename(self):
    return self._config.getstr(self, 'elmo_test_filename')
  @property
  def train_conllus(self):
    return self._config.getstr(self, 'train_conllus')
  @property
  def dev_conllus(self):
    return self._config.getstr(self, 'dev_conllus')
  @property
  def conllu_files(self):
    return self._config.getstr(self, 'conllu_files')
  @property
  def name(self):
    return self._name
  @property
  def embeddings(self):
    return self._embeddings
  @property
  def embed_placeholder(self):
    return self._embed_placeholder
  @property
  def embed_keep_prob(self):
    return self._config.getfloat(self, 'max_embed_count')
  @property
  def input_embed_keep_prob(self):
    return self._config.getfloat(self, 'input_embed_keep_prob')
  @property
  def embed_size(self):
    return self._embed_size
  @property
  def linear_size(self):
    return self._config.getint(self, 'linear_size')

# ***************************************************************
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
