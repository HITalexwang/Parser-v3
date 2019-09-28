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

#***************************************************************
# TODO maybe change self.name to something more like self._save_str?
# Ideally there should be Word2vecVocab, GloveVocab, FasttextVocab,
# each with their own _save_str
class ElmoVocab(SetVocab):
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

    super(ElmoVocab, self).__init__(config=config)
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
  def iter_sents(self, data_files):
    """"""

    for data_file in data_files:
      with codecs.open(data_file, encoding='utf-8', errors='ignore') as f:
        buff = []
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            if not re.match('[0-9]+[-.][0-9]+', line):
              buff.append(line.split('\t')[1])
          elif buff:
            yield buff
            buff = []
        if buff:
          yield buff

  #=============================================================
  def count(self, *args):
    """"""

    cur_idx = len(self.special_tokens)
    embeddings = []

    if self._elmo_test_filename:
      print ("### Loading ELMo for testset from {}! ###".format(self._elmo_test_filename))
      with h5py.File(self._elmo_test_filename, 'r') as f:
        for sid, sent in enumerate(self.iter_sents(self._conllu_files)):
          sent_ = '\t'.join(sent)
          sent_ = sent_.replace('/', '$backslash$').replace('.', '$period$')
          elmo = f[sent_].value
          assert(len(elmo) == len(sent))
          embeddings.extend(elmo)
          for wid in range(len(sent)):
            self["testset-"+str(sid)+"-"+str(wid)] = cur_idx
            cur_idx += 1
    else:
      if self._elmo_train_filename:
        print ("### Loading ELMo for trainset from {}! ###".format(self._elmo_train_filename))
        with h5py.File(self._elmo_train_filename, 'r') as f:
          for sid, sent in enumerate(self.iter_sents(self._train_conllus)):
            sent_ = '\t'.join(sent)
            sent_ = sent_.replace('/', '$backslash$').replace('.', '$period$')
            elmo = f[sent_].value
            assert(len(elmo) == len(sent))
            embeddings.extend(elmo)
            for wid in range(len(sent)):
              self["trainset-"+str(sid)+"-"+str(wid)] = cur_idx
              cur_idx += 1

      if self._elmo_dev_filename:
        print ("### Loading ELMo for devset from {}! ###".format(self._elmo_dev_filename))
        with h5py.File(self._elmo_dev_filename, 'r') as f:
          for sid, sent in enumerate(self.iter_sents(self._dev_conllus)):
            sent_ = '\t'.join(sent)
            sent_ = sent_.replace('/', '$backslash$').replace('.', '$period$')
            try:
              elmo = f[sent_].value
            except:
              raise ValueError(sent)
            assert(len(elmo) == len(sent))
            embeddings.extend(elmo)
            for wid in range(len(sent)):
              self["devset-"+str(sid)+"-"+str(wid)] = cur_idx
              cur_idx += 1

    try:
      embeddings = np.stack(embeddings)
      embeddings = np.pad(embeddings, ( (len(self.special_tokens),0), (0,0) ), 'constant')
      self._embeddings = np.stack(embeddings)
      shape = self._embeddings.shape
      self._embed_size = embeddings.shape[1]
      self._embed_placeholder = tf.placeholder(tf.float32, shape=[shape[0], shape[1]])
    except:
      shapes = set([embedding.shape for embedding in embeddings])
      raise ValueError("Couldn't stack embeddings with shapes in %s" % shapes)
    print ('### Elmo shape: {} ###'.format(self.embeddings.shape))
    return True

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

  
#***************************************************************
class FormElmoVocab(ElmoVocab, cv.FormVocab):
  pass
class LemmaElmoVocab(ElmoVocab, cv.LemmaVocab):
  pass
class UPOSElmoVocab(ElmoVocab, cv.UPOSVocab):
  pass
class XPOSElmoVocab(ElmoVocab, cv.XPOSVocab):
  pass
class DeprelElmoVocab(ElmoVocab, cv.DeprelVocab):
  pass
