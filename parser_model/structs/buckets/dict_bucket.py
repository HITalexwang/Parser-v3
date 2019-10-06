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

import numpy as np
import tensorflow as tf

from .base_bucket import BaseBucket

#***************************************************************
class DictBucket(BaseBucket):
  """"""
  
  #=============================================================
  def __init__(self, idx, depth, max_acc_depth=0, transpose_adjacency=True, config=None):
    """"""
    
    super(DictBucket, self).__init__(idx, config=config)
    
    self._depth = depth
    self._indices = []
    self._tokens = []
    self._str2idx = {}
    self._max_acc_depth = max_acc_depth
    self._transpose_adjacency = transpose_adjacency
    
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
              data[i, j, edge] = v
            else:
              data[i, j, edge] = 1
              is_arc = True
      if is_arc and self.max_acc_depth > 0:
        acc_matrices = self.accessible_matrix(data, max_acc_depth=self.max_acc_depth, 
                                                transpose=self.transpose_adjacency)
        # data[b][0] is the original graph data
        # data[b][1-max] is the accessible matrices
        data = np.concatenate([np.expand_dims(d, 1) for d in [data] + acc_matrices], 1)
        #print (data.shape)
    super(DictBucket, self).close(data)
    
    return
  
  #=============================================================
  def floyd(self, matrix):
    # Calculate the minimum distance between nodes in graph

    # matrix[b][i][j] == 1 means wj is wi's head, thus their distance is 1
    shape = matrix.shape
    seq_len = shape[1]
    dist_matrix = matrix + (1 - matrix) * 10000
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
    if transpose:
      adjacency = adjacency + np.transpose(adjacency, [0,2,1])
    dist_matrix = self.floyd(adjacency)
    # force the entries in diagonal to be 0, thus self loop distance is 0
    dist_matrix = dist_matrix * (1 - np.diag(np.ones(matrix.shape[1])).astype(int))
    #print ('dist_matrix:\n', dist_matrix)
    for d in range(max_acc_depth):
      acc_matrices.append(np.where(dist_matrix <= d+1, ones, zeros))
    return acc_matrices

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
