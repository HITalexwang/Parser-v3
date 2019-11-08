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

import os
import numpy as np
import tensorflow as tf

try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl

from .base_bucket import BaseBucket

#***************************************************************
class DictBucket(BaseBucket):
  """"""
  
  #=============================================================
  def __init__(self, idx, depth, max_acc_depth=0, transpose_adjacency=True, config=None,
                save_as_pickle=True, acc_loadname=None, top_full_connect=True,
                symmetrize_adj_first=True, insert_null_token=True):
    """"""
    
    super(DictBucket, self).__init__(idx, config=config)
    
    self._depth = depth
    self._indices = []
    self._tokens = []
    self._str2idx = {}
    self._max_acc_depth = max_acc_depth
    self._transpose_adjacency = transpose_adjacency
    self._save_as_pickle = save_as_pickle
    self._acc_loadname = acc_loadname
    self._top_full_connect = top_full_connect
    self._symmetrize_adj_first = symmetrize_adj_first
    self._insert_null_token = insert_null_token
    
    return
  
  #=============================================================
  def reset(self):
    """"""
    
    self._indices = []
    self._tokens = []
    self._str2idx = {}
    return

  #=============================================================
  def add(self, indices, tokens):
    """"""
    
    assert self._is_open, 'DictBucket is not open for adding entries'
    
    string = ' '.join(tokens)
    if string in self._str2idx:
      sequence_index = self._str2idx[string]
    else:
      sequence_index = len(self._indices)
      self._str2idx[string] = sequence_index
      self._tokens.append(tokens)
      super(DictBucket, self).add(indices)
    return sequence_index
  
  #=============================================================
  def close(self):
    """"""
    
    # Initialize the index matrix
    first_dim = len(self._indices)
    second_dim = max(len(indices) for indices in self._indices) if self._indices else 0
    shape = [first_dim, second_dim]
    if self.depth > 0:
      shape.append(self.depth)
    elif self.depth == -1:
      shape.append(shape[-1])
    
    data = np.zeros(shape, dtype=np.int32)
    # Add data to the index matrix
    if self.depth >= 0:
      try:
        for i, sequence in enumerate(self._indices):
          if sequence:
            data[i, 0:len(sequence)] = sequence
      except ValueError:
        print('Expected shape: {}\nsequence: {}'.format([len(sequence), self.depth], sequence))
        raise
    elif self.depth == -1:
      is_arc = False
      # for graphs, sequence should be list of (idx, val) pairs
      for i, sequence in enumerate(self._indices):
        for j, node in enumerate(sequence):
          for edge in node:
            if isinstance(edge, (tuple, list)):
              edge, v = edge
              if self._insert_null_token:
                data[i, j, edge+1] = v
              else:
                data[i, j, edge] = v
            else:
              if self._insert_null_token:
                data[i, j, edge+1] = 1
              else:
                data[i, j, edge] = 1
              is_arc = True
      if is_arc and self.max_acc_depth > 0:
        # save time while predicting
        if os.path.basename(self.acc_loadname).startswith('test'):
          self._acc_matrices = [data] * self.max_acc_depth
        elif self.acc_loadname and os.path.exists(self.acc_loadname):
          self.load()
        else:
          self._acc_matrices = self.accessible_matrix(data, max_acc_depth=self.max_acc_depth, 
                                                transpose=self.transpose_adjacency)
        if self.save_as_pickle:
          self.dump()
        # data[b][0] is the original graph data
        # data[b][1-max] is the accessible matrices
        data = np.concatenate([np.expand_dims(d, 1) for d in [data] + self.acc_matrices], 1)
        #print (data)
    super(DictBucket, self).close(data)
    
    return
  
  #=============================================================
  def floyd(self, matrix):
    # Calculate the minimum distance between nodes in graph

    self.d_disconnect = 10000
    # matrix[b][i][j] == 1 means wj is wi's head, thus their distance is 1
    shape = matrix.shape
    seq_len = shape[1]
    dist_matrix = matrix + (1 - matrix) * self.d_disconnect
    for b in range(shape[0]):
      for k in range(seq_len):
        for i in range(seq_len):
          for j in range(seq_len):
            if dist_matrix[b, i, k] + dist_matrix[b, k, j] < dist_matrix[b, i, j]:
              dist_matrix[b, i, j] = dist_matrix[b, i, k] + dist_matrix[b, k, j]
    return dist_matrix

  #=============================================================
  def accessible_matrix(self, matrix, max_acc_depth=0, transpose=True):
    """
    Input:
        matrix: adjacency matrix of the graph, matrix[b][i][j] == 1 means 
                wj is wi's head, thus their distance is 1
        max_acc_depth: number of accessible matrix, distance more than this 
                will be accessible in the final matrix
    """
    # Get accessible matrix with max depth

    acc_matrices = []
    ones = np.ones(matrix.shape)
    zeros = np.zeros(matrix.shape)
    adjacency = matrix
    if transpose and self.symmetrize_adj_first:
      adjacency = adjacency + np.transpose(adjacency, [0,2,1])
    dist_matrix = self.floyd(adjacency)

    diag_ones = np.diag(np.ones(matrix.shape[1])).astype(int)
    # force the entries in diagonal to be 0, thus self loop distance is 0
    if transpose and self.symmetrize_adj_first:
      dist_matrix = dist_matrix * (1 - diag_ones)
    #print ('dist_matrix:\n', dist_matrix)
    for d in range(1, max_acc_depth):
      acc_matrix = np.where(dist_matrix <= d, ones, zeros)
      if transpose and not self.symmetrize_adj_first:
        acc_matrix = (acc_matrix + np.transpose(acc_matrix, [0,2,1]))*(1 - diag_ones)+diag_ones
      acc_matrices.append(acc_matrix)
    if self.top_full_connect:
      acc_matrix = np.where(dist_matrix < self.d_disconnect, ones, zeros)
      if transpose and not self.symmetrize_adj_first:
        acc_matrix = (acc_matrix + np.transpose(acc_matrix, [0,2,1]))*(1 - diag_ones)+diag_ones
      acc_matrices.append(acc_matrix)
    else:
      acc_matrix = np.where(dist_matrix <= max_acc_depth, ones, zeros)
      if transpose and not self.symmetrize_adj_first:
        acc_matrix = (acc_matrix + np.transpose(acc_matrix, [0,2,1]))*(1 - diag_ones)+diag_ones
      acc_matrices.append(acc_matrix)
    return acc_matrices

  #=============================================================
  def dump(self):
    if self.save_as_pickle and not os.path.exists(self.acc_loadname):
      os.makedirs(os.path.dirname(self.acc_loadname), exist_ok=True)
      with open(self.acc_loadname, 'wb') as f:
        pkl.dump(self.acc_matrices, f, protocol=pkl.HIGHEST_PROTOCOL)
    return

  #=============================================================
  def load(self):
    """"""

    if self.acc_loadname and os.path.exists(self.acc_loadname):
      acc_filename = self.acc_loadname
    else:
      self._loaded = False
      return False
    print ('## Loading pre-generated accessible matrix from \'{}\''.format(self.acc_loadname))
    with open(acc_filename, 'rb') as f:
      self._acc_matrices = pkl.load(f, encoding='utf-8', errors='ignore')
    self._loaded = True
    return True


  #=============================================================
  @property
  def depth(self):
    return self._depth
  @property
  def data_indices(self):
    return self._data
  @property
  def max_acc_depth(self):
    return self._max_acc_depth
  @property
  def transpose_adjacency(self):
    return self._transpose_adjacency
  @property
  def save_as_pickle(self):
    return self._save_as_pickle
  @property
  def acc_loadname(self):
    return self._acc_loadname
  @property
  def acc_matrices(self):
    return self._acc_matrices
  @property
  def top_full_connect(self):
    return self._top_full_connect
  @property
  def symmetrize_adj_first(self):
    return self._symmetrize_adj_first
