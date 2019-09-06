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

from . import nn
from . import nonlin

DEBUG = False

#***************************************************************
def hidden(layer, hidden_size, hidden_func=nonlin.relu, hidden_keep_prob=1.):
  """"""

  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = tf.get_variable('Weights', shape=[input_size, hidden_size])#, initializer=tf.orthogonal_initializer)
  biases = tf.get_variable('Biases', shape=[hidden_size], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  layer = nn.reshape(layer, [-1, input_size])
  layer = tf.matmul(layer, weights) + biases
  layer = hidden_func(layer)
  layer = nn.reshape(layer, layer_shape + [hidden_size])
  return layer

#===============================================================
def hiddens(layer, hidden_sizes, hidden_func=nonlin.relu, hidden_keep_prob=1.):
  """"""

  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = []
  for i, hidden_size in enumerate(hidden_sizes):
    weights.append(tf.get_variable('Weights-%d' % i, shape=[input_size, hidden_size]))#, initializer=tf.orthogonal_initializer))
  weights = tf.concat(weights, axis=1)
  hidden_size = sum(hidden_sizes)
  biases = tf.get_variable('Biases', shape=[hidden_size], initializer=tf.zeros_initializer)
  if DEBUG:
    print ('### hiddens: {}, shape: {}'.format(biases.name, biases.shape))
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  layer = nn.reshape(layer, [-1, input_size])
  layer = tf.matmul(layer, weights) + biases
  layer = hidden_func(layer)
  layer = nn.reshape(layer, layer_shape + [hidden_size])
  layers = tf.split(layer, hidden_sizes, axis=-1)
  return layers

#===============================================================
def linear_classifier(layer, output_size, hidden_keep_prob=1.):
  """"""
  
  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
  biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  # (n x m x d) -> (nm x d)
  layer_reshaped = nn.reshape(layer, [-1, input_size])
  
  # (nm x d) * (d x o) -> (nm x o)
  layer = tf.matmul(layer_reshaped, weights) + biases
  # (nm x o) -> (n x m x o)
  layer = nn.reshape(layer, layer_shape + [output_size])
  return layer
  
#===============================================================
def linear_attention(layer, hidden_keep_prob=1.):
  """"""
  
  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = tf.get_variable('Weights', shape=[input_size, 1], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  # (n x m x d) -> (nm x d)
  layer_reshaped = tf.reshape(layer, [-1, input_size])
  
  # (nm x d) * (d x 1) -> (nm x 1)
  attn = tf.matmul(layer_reshaped, weights)
  # (nm x 1) -> (n x m)
  attn = tf.reshape(attn, layer_shape)
  # (n x m) -> (n x m)
  attn = tf.nn.sigmoid(attn)
  # (n x m) -> (n x 1 x m)
  soft_attn = tf.expand_dims(attn, axis=-2)
  # (n x 1 x m) * (n x m x d) -> (n x 1 x d)
  weighted_layer = tf.matmul(soft_attn, layer)
  # (n x 1 x d) -> (n x d)
  weighted_layer = tf.squeeze(weighted_layer, -2)
  return attn, weighted_layer

#===============================================================
def deep_linear_attention(layer, hidden_size, hidden_func=tf.identity, hidden_keep_prob=1.):
  """"""
  
  layer_shape = nn.get_sizes(layer)
  input_size = layer_shape.pop()
  weights = tf.get_variable('Weights', shape=[input_size, hidden_size+1], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    if len(layer_shape) > 1:
      noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    else:
      noise_shape = None
    layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
  
  # (n x m x d) -> (nm x d)
  layer_reshaped = tf.reshape(layer, [-1, input_size])
  
  # (nm x d) * (d x o+1) -> (nm x o+1)
  attn = tf.matmul(layer_reshaped, weights)
  # (nm x o+1) -> (nm x 1), (nm x o)
  attn, layer = tf.split(attn, [1, hidden_size], axis=-1)
  # (nm x 1) -> (nm x 1)
  attn = tf.nn.sigmoid(attn)
  # (nm x 1) o (nm x o) -> (nm x o)
  weighted_layer = hidden_func(layer) * attn
  # (nm x 1) -> (n x m)
  attn = tf.reshape(attn, layer_shape)
  # (nm x o) -> (n x m x o)
  weighted_layer = nn.reshape(weighted_layer, layer_shape+[hidden_size])
  return attn, weighted_layer
  
#===============================================================
def batch_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob, add_linear=True):
  """"""

  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()+add_linear
  input2_size = layer2.get_shape().as_list()[-1]+add_linear
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
    noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-add_linear])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
  if add_linear:
    ones = tf.ones(ones_shape)
    layer1 = tf.concat([layer1, ones], -1)
    layer2 = tf.concat([layer2, ones], -1)
    biases = 0
  else:
    biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
    # (o) -> (o x 1)
    biases = nn.reshape(biases, [output_size, 1])
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input1_size])
  # (n x m x d) -> (nm x d x 1)
  layer2 = nn.reshape(layer2, [-1, input2_size, 1])
  # (d x o x d) -> (d x od)
  weights = nn.reshape(weights, [input1_size, output_size*input2_size])
  
  # (nm x d) * (d x od) -> (nm x od)
  layer = tf.matmul(layer1, weights)
  # (nm x od) -> (nm x o x d)
  layer = nn.reshape(layer, [-1, output_size, input2_size])
  # (nm x o x d) * (nm x d x 1) -> (nm x o x 1)
  layer = tf.matmul(layer, layer2)
  # (nm x o x 1) -> (n x m x o)
  layer = nn.reshape(layer, layer_shape + [output_size]) + biases
  return layer

#===============================================================
def bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()+add_linear
  input2_size = layer2.get_shape().as_list()[-1]+add_linear
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.zeros_initializer)
  if DEBUG:
    print ('### bilinear classifier: {}, shape: {}'.format(weights.name, weights.shape))
  if hidden_keep_prob < 1.:
    noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
    noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-add_linear])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
  if add_linear:
    ones = tf.ones(ones_shape)
    layer1 = tf.concat([layer1, ones], -1)
    layer2 = tf.concat([layer2, ones], -1)
    biases = 0
  else:
    biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
    # (o) -> (o x 1)
    biases = nn.reshape(biases, [output_size, 1])
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input1_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input2_size])
  # (d x o x d) -> (d x od)
  weights = nn.reshape(weights, [input1_size, output_size*input2_size])
  
  # (nm x d) * (d x od) -> (nm x od)
  layer = tf.matmul(layer1, weights)
  # (nm x od) -> (n x mo x d)
  layer = nn.reshape(layer, [-1, bucket_size*output_size, input2_size])
  # (n x mo x d) * (n x m x d) -> (n x mo x m)
  layer = tf.matmul(layer, layer2, transpose_b=True)
  # (n x mo x m) -> (n x m x o x m)
  layer = nn.reshape(layer, layer_shape + [output_size, bucket_size]) + biases
  return layer

#===============================================================
def diagonal_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()
  input2_size = layer2.get_shape().as_list()[-1]
  assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
  input_size = input1_size
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if add_linear:
    weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
    weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
  biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
  
  if add_linear:
    # (n x m x d) -> (nm x d)
    lin_layer1 = nn.reshape(layer1, [-1, input_size])
    # (nm x d) * (d x o) -> (nm x o)
    lin_layer1 = tf.matmul(lin_layer1, weights1)
    # (nm x o) -> (n x m x o)
    lin_layer1 = nn.reshape(lin_layer1, layer_shape + [output_size])
    # (n x m x o) -> (n x m x o x 1)
    lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
    # (n x m x d) -> (nm x d)
    lin_layer2 = nn.reshape(layer2, [-1, input_size])
    # (nm x d) * (d x o) -> (nm x o)
    lin_layer2 = tf.matmul(lin_layer2, weights1)
    # (nm x o) -> (n x m x o)
    lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
    # (n x m x o) -> (n x o x m)
    lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
    # (n x o x m) -> (n x 1 x o x m)
    lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)
  
  # (n x m x d) -> (n x m x 1 x d)
  layer1 = nn.reshape(layer1, [-1, bucket_size, 1, input_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
  # (d x o) -> (o x d)
  weights = tf.transpose(weights, [1, 0])
  # (o) -> (o x 1)
  biases = nn.reshape(biases, [output_size, 1])
  
  # (n x m x 1 x d) (*) (o x d) -> (n x m x o x d)
  layer = layer1 * weights
  # (n x m x o x d) -> (n x mo x d)
  layer = nn.reshape(layer, [-1, bucket_size*output_size, input_size])
  # (n x mo x d) * (n x m x d) -> (n x mo x m)
  layer = tf.matmul(layer, layer2, transpose_b=True)
  # (n x mo x m) -> (n x m x o x m)
  layer = nn.reshape(layer, layer_shape + [output_size, bucket_size])
  if add_linear:
    # (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
    layer += lin_layer1 + lin_layer2
  # (n x m x o x m) + (o x 1) -> (n x m x o x m)
  layer += biases
  return layer

#===============================================================
def bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()+1
  input2_size = layer2.get_shape().as_list()[-1]+1
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if hidden_keep_prob < 1.:
    noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
    noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
  ones = tf.ones(ones_shape)
  layer1 = tf.concat([layer1, ones], -1)
  layer2 = tf.concat([layer2, ones], -1)
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input1_size])
  # (n x m x d) -> (n x m x d)
  layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
  
  # (nm x d) * (d x d) -> (nm x d)
  layer = tf.matmul(layer1, weights)
  # (nm x d) -> (n x m x d)
  layer = nn.reshape(layer, [-1, bucket_size, input2_size])
  # (n x m x d) * (n x m x d) -> (n x m x m)
  layer = tf.matmul(layer, layer2, transpose_b=True)
  # (n x mo x m) -> (n x m x m)
  layer = nn.reshape(layer, layer_shape + [bucket_size])
  return layer

#===============================================================
def diagonal_bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()
  input2_size = layer2.get_shape().as_list()[-1]
  assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
  input_size = input1_size
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if add_linear:
    weights1 = tf.get_variable('Weights1', shape=[input_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
    weights2 = tf.get_variable('Weights2', shape=[input_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
  biases = tf.get_variable('Biases', shape=[1], initializer=tf.zeros_initializer)
  if hidden_keep_prob < 1.:
    noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
  
  if add_linear:
    #(d) -> (d x 1)
    weights1 = tf.expand_dims(weights1, axis=-1)
    # (n x m x d) -> (nm x d)
    lin_layer1 = nn.reshape(layer1, [-1, input_size])
    # (nm x d) * (d x 1) -> (nm x 1)
    lin_layer1 = tf.matmul(lin_layer1, weights1)
    # (nm x 1) -> (n x m)
    lin_layer1 = nn.reshape(lin_layer1, layer_shape)
    # (n x m) -> (n x m x 1)
    lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
    #(d) -> (d x 1)
    weights2 = tf.expand_dims(weights2, axis=-1)
    # (n x m x d) -> (nm x d)
    lin_layer2 = nn.reshape(layer2, [-1, input_size])
    # (nm x d) * (d x 1) -> (nm x 1)
    lin_layer2 = tf.matmul(lin_layer2, weights1)
    # (nm x 1) -> (n x m)
    lin_layer2 = nn.reshape(lin_layer2, layer_shape)
    # (n x m) -> (n x 1 x m)
    lin_layer2 = tf.expand_dims(lin_layer2, axis=-2)
  
  # (n x m x d) -> (n x m x d)
  layer1 = nn.reshape(layer1, [-1, bucket_size, input_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
  
  # (n x m x d) (*) (d) -> (n x m x d)
  layer = layer1 * weights
  # (n x m x d) * (n x m x d) -> (n x m x m)
  layer = tf.matmul(layer, layer2, transpose_b=True)
  # (n x m x m) -> (n x m x m)
  layer = nn.reshape(layer, layer_shape + [bucket_size])
  if add_linear:
    # (n x m x m) + (n x 1 x m) + (n x m x 1) -> (n x m x m)
    layer += lin_layer1 + lin_layer2
  # (n x m x m) + () -> (n x m x m)
  layer += biases
  return layer

#===============================================================
def bilinear_attention(layer1, layer2, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()+add_linear
  input2_size = layer2.get_shape().as_list()[-1]
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
  if DEBUG:
    print ('### bilinear attention: {}, shape: {}'.format(weights.name, weights.shape))
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  original_layer1 = layer1
  if hidden_keep_prob < 1.:
    noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
    noise_shape2 = tf.stack(layer_shape[:-2] + [1, input2_size])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
  if add_linear:
    ones = tf.ones(ones_shape)
    layer1 = tf.concat([layer1, ones], -1)
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input1_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input2_size])
  
  # (nm x d) * (d x d) -> (nm x d)
  attn = tf.matmul(layer1, weights)
  # (nm x d) -> (n x m x d)
  attn = nn.reshape(attn, [-1, bucket_size, input2_size])
  # (n x m x d) * (n x m x d) -> (n x m x m)
  attn = tf.matmul(attn, layer2, transpose_b=True)
  # (n x m x m) -> (n x m x m)
  attn = nn.reshape(attn, layer_shape + [bucket_size])
  # (n x m x m) -> (n x m x m)
  soft_attn = tf.nn.softmax(attn)
  # (n x m x m) * (n x m x d) -> (n x m x d)
  weighted_layer1 = tf.matmul(soft_attn, original_layer1)
  
  return attn, weighted_layer1
  
#===============================================================
def diagonal_bilinear_attention(layer1, layer2, hidden_keep_prob=1., add_linear=True):
  """"""
  
  layer_shape = nn.get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()
  input2_size = layer2.get_shape().as_list()[-1]
  assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
  input_size = input1_size
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  if add_linear:
    weights2 = tf.get_variable('Weights2', shape=[input_size], initializer=tf.zeros_initializer)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
  original_layer1 = layer1
  if hidden_keep_prob < 1.:
    noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
    layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
    layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
  
  if add_linear:
    #(d) -> (d x 1)
    weights2 = tf.expand_dims(weights2, axis=-1)
    # (n x m x d) -> (nm x d)
    lin_attn2 = nn.reshape(layer2, [-1, input_size])
    # (nm x d) * (d x 1) -> (nm x 1)
    lin_attn2 = tf.matmul(lin_attn2, weights2)
    # (nm x 1) -> (n x m)
    lin_attn2 = nn.reshape(lin_attn2, layer_shape)
    # (n x m) -> (n x 1 x m)
    lin_attn2 = tf.expand_dims(lin_attn2, axis=-2)
  
  # (n x m x d) -> (nm x d)
  layer1 = nn.reshape(layer1, [-1, input_size])
  # (n x m x d) -> (n x m x d)
  layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
  
  # (nm x d) * (d) -> (nm x d)
  attn = layer1 * weights
  # (nm x d) -> (n x m x d)
  attn = nn.reshape(attn, [-1, bucket_size, input_size])
  # (n x m x d) * (n x m x d) -> (n x m x m)
  attn = tf.matmul(attn, layer2, transpose_b=True)
  # (n x m x m) -> (n x m x m)
  attn = nn.reshape(attn, layer_shape + [bucket_size])
  if add_linear:
    # (n x m x m) + (n x 1 x m) -> (n x m x m)
    attn += lin_attn2
  # (n x m x m) -> (n x m x m)
  soft_attn = tf.nn.softmax(attn)
  # (n x m x m) * (n x m x d) -> (n x m x d)
  weighted_layer1 = tf.matmul(soft_attn, original_layer1)

  return attn, weighted_layer1


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.losses.losses_impl import Reduction, compute_weighted_loss

def sigmoid_cross_entropy(
    multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None, fp_cost=.0, fn_cost=.0,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.
  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.
  If `label_smoothing` is nonzero, smooth the labels towards 1/2:
      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing
  Args:
    multi_class_labels: `[batch_size, num_classes]` target integer labels in
      `{0, 1}`.
    logits: Float `[batch_size, num_classes]` logits outputs of the network.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    label_smoothing: If greater than `0` then smooth the labels.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has the same shape as `logits`; otherwise, it is scalar.
  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `multi_class_labels` or if the shape of `weights` is invalid, or if
      `weights` is None.  Also if `multi_class_labels` or `logits` is None.
  @compatbility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if multi_class_labels is None:
    raise ValueError("multi_class_labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "sigmoid_cross_entropy_loss",
                      (logits, multi_class_labels, weights)) as scope:
    logits = ops.convert_to_tensor(logits)
    multi_class_labels = math_ops.cast(multi_class_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(multi_class_labels.get_shape())

    if label_smoothing > 0:
      multi_class_labels = (multi_class_labels * (1 - label_smoothing) +
                            0.5 * label_smoothing)

    losses = sigmoid_cross_entropy_with_logits(labels=multi_class_labels,
                                                  logits=logits,
                                                  name="xentropy",
                                                  fp_cost=fp_cost,
                                                  fn_cost=fn_cost)
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def sigmoid_cross_entropy_with_logits(  # pylint: disable=invalid-name
    _sentinel=None,
    labels=None,
    logits=None,
    name=None,
    fp_cost=None,
    fn_cost=None):
  """Computes sigmoid cross entropy given `logits`.
  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.  For instance, one could
  perform multilabel classification where a picture can contain both an elephant
  and a dog at the same time.
  For brevity, let `x = logits`, `z = labels`.  The logistic loss is
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))
  For x < 0, to avoid overflow in exp(-x), we reformulate the above
        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))
  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation
      max(x, 0) - x * z + log(1 + exp(-abs(x)))
  `logits` and `labels` must have the same type and shape.
  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.
  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  # pylint: disable=protected-access
  nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
                           labels, logits)
  # pylint: enable=protected-access

  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.

    # With fp/fn loss:
    # For x > 0
    # x - x * z + z * log(1 + exp(-x + cost_fn)) + (1 - z) * log(1 + exp(-x + cost_fp))
    # For x < 0
    # - x * z + z * log(exp(x) + exp(cost_fn)) + (1 - z) * log(exp(x) + exp(cost_fp))
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    ones = array_ops.ones_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    #print (cond)
    #relu_logits = array_ops.where(cond, logits, zeros)
    #neg_abs_logits = array_ops.where(cond, -logits, logits)
    fn_cond = math_ops.logical_and(labels > zeros, logits < zeros)
    fp_cond = math_ops.logical_and(labels <= zeros, logits >= zeros)
    cost_fn = fn_cost * math_ops.cast(fn_cond, dtypes.float32)
    cost_fp = fp_cost * math_ops.cast(fp_cond, dtypes.float32)

    pos_loss = logits - logits * labels + labels * math_ops.log1p(math_ops.exp(-logits + cost_fn)) + (
      (ones - labels) * math_ops.log1p(math_ops.exp(-logits + cost_fp)))
    neg_loss = - logits * labels + labels * math_ops.log(math_ops.exp(logits) + math_ops.exp(cost_fn)) + (
      (ones - labels) * math_ops.log(math_ops.exp(logits) + math_ops.exp(cost_fp)))

    return array_ops.where(cond, pos_loss, neg_loss, name=name)