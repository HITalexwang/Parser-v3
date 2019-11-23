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
try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl
  
import curses
import time

import numpy as np
import tensorflow as tf

from parser_model.neural import nn
from scripts.chuliu_edmonds import chuliu_edmonds_one_root
from scripts.mst import nonprojective

#***************************************************************
class GraphOutputs(object):
  """"""
  
  _dataset = None
  
  _print_mapping = [('form', 'Form'),
                    ('lemma', 'Lemma'),
                    ('upos', 'UPOS'),
                    ('xpos', 'XPOS'),
                    ('ufeats', 'UFeat'),
                    ('dephead', 'UAS'),
                    ('deprel', 'OLS'),
                    ('deptree', 'LAS'),
                    ('semhead', 'UF1'),
                    ('semrel', 'OLS'),
                    ('semgraph', 'LF1')]
  
  #=============================================================
  def __init__(self, outputs, tokens, load=False, evals=None, factored_deptree=None, 
                factored_semgraph=None, config=None, loss_interpolation=.5):
    """"""
    
    self._factored_deptree = factored_deptree
    self._factored_semgraph = factored_semgraph
    self._config = config
    self._evals = evals or list(outputs.keys())
    #self._evals = config.getlist(self, 'evals')
    valid_evals = set([print_map[0] for print_map in self._print_mapping])
    self._accuracies = {'total': tokens}
    self._probabilities = {}
    self.time = None

    self._loss_interpolation = loss_interpolation
    
    # Store the predicted graph matrix at each attention layer
    if 'acc' in outputs:
      acc_output = outputs.pop('acc')
      self._probabilities['acc'] = acc_output['probabilities']
      if 'acc' in self._evals:
        self._evals.remove('acc')

    self._losses_by_layer = []
    if 'semgraph' in outputs:
      if 'losses_by_layer' in outputs['semgraph'] and 'label_loss' in outputs['semgraph']:
        rho = self.loss_interpolation
        labeled_loss = outputs['semgraph']['label_loss']
        for layer_idx, unlabeled_loss in enumerate(outputs['semgraph']['losses_by_layer']):
          with tf.variable_scope("loss_of_layer_%d" % layer_idx):
            loss = 2*((1-rho) * unlabeled_loss + rho * labeled_loss)
            self._losses_by_layer.append(loss)

    for eval_ in list(self._evals):
      assert eval_ in valid_evals
    self._loss = tf.add_n([tf.where(tf.is_finite(output['loss']), output['loss'], 0.) for output in outputs.values()])
    
    #-----------------------------------------------------------
    for field in outputs:
      self._probabilities[field] = outputs[field].pop('probabilities')
      if 'unlabeled_probabilities' in outputs[field]:
        self._probabilities['semhead'] = outputs[field].pop('unlabeled_probabilities')
      if 'label_probabilities' in outputs[field]:
        self._probabilities['semrel'] = outputs[field].pop('label_probabilities')
      self._accuracies[field] = outputs[field]

    #-----------------------------------------------------------
    filename = os.path.join(self.save_dir, '{}.pkl'.format(self.dataset))
    # TODO make a separate History object
    if load and os.path.exists(filename):
      with open(filename, 'rb') as f:
        self.history = pkl.load(f)
    else:
      self.history = {
        'total': {'n_batches' : 0,
                  'n_tokens': 0,
                  'n_sequences': 0,
                  'total_time': 0},
        'speed': {'toks/sec': [],
                  'seqs/sec': [],
                  'bats/sec': []}
        }
      for field in self._accuracies:
        if field == 'semgraph':
          for string in ('head', 'graph'):
            self.history['sem'+string] = {
              'loss': [0],
              'tokens': [0],
              'fp_tokens': 0,
              'fn_tokens': 0,
              'sequences': [0]
            }
          if self._factored_semgraph:
            self.history['semrel'] = {
              'loss': [0],
              'tokens': [0],
              'n_edges': 0,
              'sequences': [0]
            }
          if 'n_acc_true_positives' in self._accuracies[field]:
            if isinstance(self._accuracies[field]['n_acc_true_positives'], list):
              n_layers = len(self._accuracies[field]['n_acc_true_positives'])
              self.history['accessible'] = {
                'loss': [0],
                'tokens': [[0]*n_layers],
                'fp_tokens': [0]*n_layers,
                'fn_tokens': [0]*n_layers
              } 
            else:
              self.history['accessible'] = {
                'loss': [0],
                'tokens': [0],
                'fp_tokens': 0,
                'fn_tokens': 0
              }
          elif 'acc_loss' in self._accuracies[field]:
            self.history['accessible'] = {
              'loss': [0]
            }
            
        elif field == 'deptree':
          for string in ('head', 'tree'):
            self.history['dep'+string] = {
              'loss': [0],
              'tokens': [0],
              'sequences': [0]
            }
          if self._factored_deptree:
            self.history['deprel'] = {
              'loss': [0],
              'tokens': [0],
              'sequences': [0]
            }
        elif field not in ('speed', 'total'):
          self.history[field] ={
            'loss': [0],
            'tokens': [0],
            'sequences': [0]
          }
    self.predictions = {'indices': []}
    return
  
  #=============================================================
  def probs_to_preds(self, probabilities, lengths, augment_layers=None, policy='confidence',
                      predict_rel_in_attention=False):
    """"""

    predictions = {}
    
    if 'form' in probabilities:
      form_probs = probabilities['form']
      if isinstance(form_probs, (tuple, list)):
        form_samples, form_probs = form_probs
        form_preds = np.argmax(form_probs, axis=-1)
        predictions['form'] = form_samples[np.arange(len(form_preds)), form_preds]
      else:
        form_preds = np.argmax(form_probs, axis=-1)
        predictions['form'] = form_preds
    if 'lemma' in probabilities:
      lemma_probs = probabilities['lemma']
      lemma_preds = np.argmax(lemma_probs, axis=-1)
      predictions['lemma'] = lemma_preds
    if 'upos' in probabilities:
      upos_probs = probabilities['upos']
      upos_preds = np.argmax(upos_probs, axis=-1)
      predictions['upos'] = upos_preds
    if 'xpos' in probabilities:
      xpos_probs = probabilities['xpos']
      if isinstance(xpos_probs, (tuple, list)):
        xpos_preds = np.concatenate([np.argmax(xpos_prob_mat, axis=-1)[:,:,None] for xpos_prob_mat in xpos_probs], axis=-1)
      else:
        xpos_preds = np.argmax(xpos_probs, axis=-1)
      predictions['xpos'] = xpos_preds
    if 'ufeats' in probabilities:
      ufeats_probs = probabilities['ufeats']
      ufeats_preds = np.concatenate([np.argmax(ufeats_prob_mat, axis=-1)[:,:,None] for ufeats_prob_mat in ufeats_probs], axis=-1)
      predictions['ufeats'] = ufeats_preds
    #if 'head' in probabilities: # TODO MST algorithms
    #  head_probs = probabilities['head']
    #  head_preds = np.argmax(head_probs, axis=-1)
    #  predictions['head'] = head_preds
    if 'deptree' in probabilities:
      # (n x m x m x c)
      deptree_probs = probabilities['deptree']
      if self._factored_deptree:
        # (n x m x m x c) -> (n x m x m)
        dephead_probs = deptree_probs.sum(axis=-1)
        # (n x m x m) -> (n x m)
        #dephead_preds = np.argmax(dephead_probs, axis=-1)
        dephead_preds = np.zeros(dephead_probs.shape[:2], dtype=np.int32)
        for i, (_dephead_probs, length) in enumerate(zip(dephead_probs, lengths)):
          #print(_dephead_probs)
          #input()
          #cle = chuliu_edmonds_one_root(_dephead_probs[:length, :length])
          cle = nonprojective(_dephead_probs[:length, :length])
          dephead_preds[i, :length] = cle
        # ()
        bucket_size = dephead_preds.shape[1]
        # (n x m) -> (n x m x m)
        one_hot_dephead_preds = (np.arange(bucket_size) == dephead_preds[...,None]).astype(int)
        # (n x m x m) * (n x m x m x c) -> (n x m x c)
        deprel_probs = np.einsum('ijk,ijkl->ijl', one_hot_dephead_preds, deptree_probs)
        # (n x m x c) -> (n x m)
        deprel_preds = np.argmax(deprel_probs, axis=-1)
      else:
        # (), ()
        bucket_size, n_classes = deptree_probs.shape[-2:]
        # (n x m x m x c) -> (n x m x mc)
        deptree_probs = deptree_probs.reshape([-1, bucket_size, bucket_size*n_classes])
        # (n x m x mc) -> (n x m)
        deptree_preds = np.argmax(deptree_probs, axis=-1)
        # (n x m) -> (n x m)
        dephead_preds = deptree_preds // bucket_size
        deprel_preds = deptree_preds % n_classes
      predictions['dephead'] = dephead_preds
      predictions['deprel'] = deprel_preds
    if 'semgraph' in probabilities:
      # (n x m x m x c)
      semgraph_probs = probabilities['semgraph']
      if self._factored_semgraph and self.decoder == 'sem16':
        #print ('### Decoder:sem16 ###')
        semgraph_preds = self.sem16decoder(semgraph_probs, lengths)
      elif self._factored_semgraph and self.decoder == 'easy-first':
        assert 'semhead' in probabilities
        if predict_rel_in_attention:
          assert 'semrel' in probabilities
          predictions['semrel'] = self.easyfirst_rel_decoder(probabilities['semhead'], probabilities['semrel'], lengths)
        else:
          predictions['semrel'] = self.easyfirst_decoder(probabilities['semhead'], semgraph_probs, lengths,
                                                        policy=policy)
        predictions['semhead'] = []
        return predictions
      elif self._factored_semgraph:
        # (n x m x m x c) -> (n x m x m)
        semhead_probs = semgraph_probs.sum(axis=-1)
        # (n x m x m) -> (n x m x m)
        semhead_preds = np.where(semhead_probs >= .5, 1, 0)
        if augment_layers is not None:
          semhead_preds = self.augment_head_with_acc(semhead_preds, probabilities['acc'], augment_layers)
        # (n x m x m x c) -> (n x m x m)
        semrel_preds = np.argmax(semgraph_probs, axis=-1)
        # (n x m x m) (*) (n x m x m) -> (n x m x m)
        semgraph_preds = semhead_preds * semrel_preds
      else:
        # (n x m x m x c) -> (n x m x m)
        semgraph_preds = np.argmax(semgraph_probs, axis=-1)
      predictions['semrel'] = sparse_semgraph_preds = []
      predictions['semhead'] = []
      for i in range(len(semgraph_preds)):
        sparse_semgraph_preds.append([])
        for j in range(len(semgraph_preds[i])):
          sparse_semgraph_preds[-1].append([])
          for k, pred in enumerate(semgraph_preds[i,j]):
            if pred:
              sparse_semgraph_preds[-1][-1].append((k, semgraph_preds[i,j,k]))
    return predictions

  def easyfirst_rel_decoder(self, semhead_probs, semrel_probs, lengths):
    #print (len(semhead_probs), len(semhead_probs[0]), semhead_probs[0][0].shape)
    # remove the null token
    seq_len = semrel_probs[0].shape[-2] - 1
    batch_size = semrel_probs[0].shape[0]
    sparse_semgraph_preds = [[[] for _ in range(seq_len)] 
                                  for _ in range(batch_size)]
    max_layer = self.num_used_layers if self.num_used_layers > 0 else len(semhead_probs)
    print ("Using the base {} layers for predicting.".format(max_layer))
    # collect heads from each layer and each attention head
    # for each attention layer
    for n_layer, (head_probs, rel_probs) in enumerate(zip(semhead_probs,semrel_probs)):
      if n_layer >= max_layer: break
      # (n x m x m x c) -> (n x m x m)
      semrel_preds = np.argmax(rel_probs, axis=-1)[:,1:,:]
      #print ('semrel_preds:\n',semrel_preds)
      # for each attention head
      for h, probs in enumerate(head_probs):
        # remove the null token at 0 
        head_indices = np.argmax(probs, axis=-1)[:,1:]
        #print (probs)
        #print (head_indices)
        # the i-th sentence in the batch
        for i, heads in enumerate(head_indices):
          for j, head in enumerate(heads):
            if head > 0:
              head_exists = False
              # do not add duplicate arc
              for h_, s in sparse_semgraph_preds[i][j]:
                if h_ == head-1:
                  head_exists = True
                  break
              if not head_exists:
                # substract the index of null token, so null token == -1
                sparse_semgraph_preds[i][j].append((head-1,semrel_preds[i,j,head]))
    #print (sparse_semgraph_preds)
    return sparse_semgraph_preds

  def easyfirst_decoder(self, semhead_probs, semrel_probs, lengths, policy='confidence'):
    #print (len(semhead_probs), len(semhead_probs[0]), semhead_probs[0][0].shape)
    #print (semrel_probs.shape)

    # (n x m x m x c) -> (n x m x m)
    semrel_preds = np.argmax(semrel_probs, axis=-1)[:,1:,:]
    # remove the null token
    seq_len = semrel_preds.shape[-1] - 1
    sparse_semgraph_preds = [[[] for _ in range(seq_len)] 
                                  for _ in range(semrel_preds.shape[0])]
    # collect heads from each layer and each attention head
    # for each attention layer
    for n_layer, headprobs in enumerate(semhead_probs):
      # for each attention head
      for h, probs in enumerate(headprobs):
        if policy == 'top_k':
          # (n x m x m)
          # remove the null token
          semhead_preds = np.where(probs >= .5, 1, 0)[:,1:,:]
          # (n x m x m) (*) (n x m x m) -> (n x m x m)
          semgraph_preds = semhead_preds * semrel_preds
          # i-th sentence
          for i in range(len(semgraph_preds)):
            # j-th word
            for j in range(len(semgraph_preds[i])):
              # k-th word
              for k, pred in enumerate(semgraph_preds[i,j]):
                if pred:
                  head_exists = False
                  for h, s in sparse_semgraph_preds[i][j]:
                    if h == k-1:
                      head_exists = True
                      break
                  if not head_exists:
                    # substract the index of null token
                    sparse_semgraph_preds[i][j].append((k-1, semgraph_preds[i,j,k]))
        else:
          # remove the null token at 0 
          head_indices = np.argmax(probs, axis=-1)[:,1:]
          #print (probs)
          #print (head_indices)
          # the i-th sentence in the batch
          for i, heads in enumerate(head_indices):
            for j, head in enumerate(heads):
              if head > 0:
                head_exists = False
                # do not add duplicate arc
                for h_, s in sparse_semgraph_preds[i][j]:
                  if h_ == head-1:
                    head_exists = True
                    break
                if not head_exists:
                  # substract the index of null token, so null token == -1
                  sparse_semgraph_preds[i][j].append((head-1,semrel_preds[i,j,head]))
    #print (sparse_semgraph_preds)
    return sparse_semgraph_preds

  def augment_head_with_acc(self, semhead_preds, acc_probs, n_layers=[1]):
    """
    Augment semhead_preds with arcs predicted in graph attention masks
    Input:
          semhead_preds: shape: (n x m x m), 0-1 matrix for output graph
          n_layers: the layers which are used to augment the output graph
    """
    output_preds = semhead_preds
    #n_addeds = []
    aug_preds = np.ones_like(semhead_preds)
    #print ('orig:\n',output_preds)
    for n in n_layers:
      probs = acc_probs[n]
      # (n x m x m) -> (n x m x m)
      preds = np.where(probs >= .5, 1, 0)
      #print ('layer-{}:\n'.format(n),preds)
      aug_preds = aug_preds * preds
    #print ('aug_preds:\n',aug_preds)
    overlapped = aug_preds * output_preds
    n_added = np.sum(aug_preds - overlapped)
    #n_addeds.append(n_added)
    #print ("## Added {} arcs from layer-{}.".format(n_added, n))
    output_preds = output_preds + aug_preds - overlapped
    print ('## Added {} arcs in total.'.format(n_added))
    return output_preds
  
  def sem16decoder(self, semgraph_probs, lengths):
    # (n x m x m x c) -> (n x m x m)
    semhead_probs = semgraph_probs.sum(axis=-1)
    # (n x m x m) -> (n x m x m)
    semhead_preds = np.where(semhead_probs >= .5, 1, 0)
    # (n x m x m)
    masked_semhead_preds = np.zeros(semhead_preds.shape, dtype=np.int32)
    # mask by length
    for i, (sem_preds, length) in enumerate(zip(semhead_preds, lengths)):
      masked_semhead_preds[i,:length,:length] = sem_preds[:length,:length]
    n_counts = {'no_root':0, 'multi_root':0, 'no_head':0, 'self_circle':0}
    # for each sentence
    #for i in range(len(masked_semhead_preds)):
    for i, length in enumerate(lengths):
      for j in range(length):
        if masked_semhead_preds[i,j,j] == 1:
          #print ('self circle line:',j,'\n',masked_semhead_preds[i])
          n_counts['self_circle'] += 1
          masked_semhead_preds[i,j,j] = 0
          #print ('new graph:\n',masked_semhead_preds[i])
      n_root = np.sum(masked_semhead_preds[i,:,0])
      if n_root == 0:
        #print ('root:', n_root, '\n',masked_semhead_preds[i])
        n_counts['no_root'] += 1
        new_root = np.argmax(semhead_probs[i,1:,0]) + 1
        masked_semhead_preds[i,new_root,0] = 1
        #print ('new graph:\n', masked_semhead_preds[i])
      elif n_root > 1:
        #print ('root:', n_root, '\n',masked_semhead_preds[i])
        n_counts['multi_root'] += 1
        kept_root = np.argmax(semhead_probs[i,1:,0]) + 1
        masked_semhead_preds[i,:,0] = 0
        masked_semhead_preds[i,kept_root,0] = 1
        #print ('new graph:\n', masked_semhead_preds[i])
      n_heads = masked_semhead_preds[i,:length,:length].sum(axis=-1)
      # no need to check line 0
      n_heads[0] = 1
      for j, n_head in enumerate(n_heads):
        if n_head == 0:
          #print ('no head line:',j,'\n',masked_semhead_preds[i])
          n_counts['no_head'] += 1
          # make sure the new head is not self circle
          semhead_probs[i,j,j] = 0
          new_head = np.argmax(semhead_probs[i,j,1:length]) + 1
          masked_semhead_preds[i,j,new_head] = 1
          #print ('new graph:\n',masked_semhead_preds[i])
    #print ('Corrected List:','\t'.join([key+':'+str(val) for key,val in n_counts.items()]))
    # (n x m x m x c) -> (n x m x m)
    semrel_preds = np.argmax(semgraph_probs, axis=-1)
    # (n x m x m) (*) (n x m x m) -> (n x m x m)
    semgraph_preds = masked_semhead_preds * semrel_preds

    return semgraph_preds

  #=============================================================
  def cache_predictions(self, tokens, indices):
    """"""
    
    self.predictions['indices'].extend(indices)
    for field in tokens:
      if field not in self.predictions:
        self.predictions[field] = []
      self.predictions[field].extend(tokens[field])
    return
  
  #=============================================================
  def print_current_predictions(self):
    """"""
    
    order = np.argsort(self.predictions['indices'])
    fields = ['form', 'lemma', 'upos', 'xpos', 'ufeats', 'dephead', 'deprel', 'semrel', 'misc']
    for i in order:
      j = 1
      token = []
      while j < len(self.predictions['id'][i]):
        token = [self.predictions['id'][i][j]]
        for field in fields:
          if field in self.predictions:
            token.append(self.predictions[field][i][j])
          else:
            token.append('_')
        print(u'\t'.join(token))
        j += 1
      print('')
    self.predictions = {'indices': []}
    return

  #=============================================================
  def dump_current_predictions(self, f):
    """"""
    
    order = np.argsort(self.predictions['indices'])
    fields = ['form', 'lemma', 'upos', 'xpos', 'ufeats', 'dephead', 'deprel', 'semrel', 'misc']
    for i in order:
      j = 1
      token = []
      while j < len(self.predictions['id'][i]):
        token = [self.predictions['id'][i][j]]
        for field in fields:
          if field in self.predictions:
            token.append(self.predictions[field][i][j])
          else:
            token.append('_')
        f.write('\t'.join(token)+'\n')
        j += 1
      f.write('\n')
    self.predictions = {'indices': []}
    return

  #=============================================================
  def get_current_predictions(self):
    """"""
    
    order = np.argsort(self.predictions['indices'])
    fields = ['form', 'lemma', 'upos', 'xpos', 'ufeats', 'dephead', 'deprel', 'semrel', 'misc']
    predictions = []
    for i in order:
      j = 1
      token = []
      tokens = []
      while j < len(self.predictions['id'][i]):
        token = [self.predictions['id'][i][j]]
        for field in fields:
          if field in self.predictions:
            token.append(self.predictions[field][i][j])
          else:
            token.append('_')
        tokens.append(token)
        j += 1
      predictions.append(tokens)
    self.predictions = {'indices': []}
    return predictions
  
  #=============================================================
  def compute_token_accuracy(self, field):
    """"""
    
    return self.history[field]['tokens'][-1] / (self.history['total']['n_tokens'] + 1e-12)
  
  def compute_token_F1(self, field):
    """"""
    
    precision = self.history[field]['tokens'][-1] / (self.history[field]['tokens'][-1] + self.history[field]['fp_tokens'] + 1e-12)
    recall = self.history[field]['tokens'][-1] / (self.history[field]['tokens'][-1] + self.history[field]['fn_tokens'] + 1e-12)
    return 2 * (precision * recall) / (precision + recall + 1e-12)

  def compute_token_F1_PR(self, field):
    """"""
    
    precision = self.history[field]['tokens'][-1] / (self.history[field]['tokens'][-1] + self.history[field]['fp_tokens'] + 1e-12)
    recall = self.history[field]['tokens'][-1] / (self.history[field]['tokens'][-1] + self.history[field]['fn_tokens'] + 1e-12)
    return 2 * (precision * recall) / (precision + recall + 1e-12), precision, recall

  def compute_token_F1_PR_index(self, field, i):
    """"""
    
    precision = self.history[field]['tokens'][-1][i] / (self.history[field]['tokens'][-1][i] + self.history[field]['fp_tokens'][i] + 1e-12)
    recall = self.history[field]['tokens'][-1][i] / (self.history[field]['tokens'][-1][i] + self.history[field]['fn_tokens'][i] + 1e-12)
    return 2 * (precision * recall) / (precision + recall + 1e-12), precision, recall
  
  def compute_sequence_accuracy(self, field):
    """"""
    
    return self.history[field]['sequences'][-1] / self.history['total']['n_sequences']
  
  #=============================================================
  def get_current_accuracy(self):
    """"""
    
    token_accuracy = 0
    for field in self.history:
      if field in self.evals:
        if field.startswith('sem'):
          token_accuracy += np.log(self.compute_token_F1(field)+1e-12)
        else:
          token_accuracy += np.log(self.compute_token_accuracy(field)+1e-12)
    token_accuracy /= len(self.evals)
    return np.exp(token_accuracy) * 100
  
  #=============================================================
  def get_current_geometric_accuracy(self):
    """"""
    
    token_accuracy = 0
    for field in self.history:
      if field in self.evals:
        if field.startswith('sem'):
          token_accuracy += np.log(self.compute_token_F1(field)+1e-12)
        else:
          token_accuracy += np.log(self.compute_token_accuracy(field)+1e-12)
    token_accuracy /= len(self.evals)
    return np.exp(token_accuracy) * 100
  
  #=============================================================
  def restart_timer(self):
    """"""
    
    self.time = time.time()
    return
  
  #=============================================================
  def update_history(self, outputs, show=False):
    """"""
    #print ('remained heads:\n',outputs['semgraph']['allowed_heads'])
    #print ('used heads(y):\n',outputs['semgraph']['used_heads'])
    if show:
      import numpy
      numpy.set_printoptions(threshold=numpy.nan)
      print ('remained heads:\n',outputs['semgraph']['allowed_heads'])
      print ('used heads(y):\n',outputs['semgraph']['used_heads'])
      print ('unlabeled by layer:\n',outputs['semgraph']['unlabeled_by_layer'])
      print ('label by layer:\n',outputs['semgraph']['label_by_layer'])
      print ('sumed unlabeled:\n',outputs['semgraph']['unlabeled_predictions'])
      print ('sumed label:\n',outputs['semgraph']['label_predictions'])

    self.history['total']['total_time'] += time.time() - self.time
    self.time = None
    self.history['total']['n_batches'] += 1
    self.history['total']['n_tokens'] += outputs['total']['n_tokens']
    self.history['total']['n_sequences'] += outputs['total']['n_sequences']
    for field, output in six.iteritems(outputs):
      if field == 'semgraph':
        if self._factored_semgraph:
          self.history['semrel']['loss'][-1] += output['label_loss']
          self.history['semrel']['tokens'][-1] += output['n_correct_label_tokens']
          self.history['semrel']['n_edges'] += output['n_true_positives'] + output['n_false_negatives']
          self.history['semrel']['sequences'][-1] += output['n_correct_label_sequences']
        self.history['semhead']['loss'][-1] += output['unlabeled_loss']
        self.history['semhead']['tokens'][-1] += output['n_unlabeled_true_positives']
        self.history['semhead']['fp_tokens'] += output['n_unlabeled_false_positives']
        self.history['semhead']['fn_tokens'] += output['n_unlabeled_false_negatives']
        self.history['semhead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
        self.history['semgraph']['loss'][-1] += output['loss']
        self.history['semgraph']['tokens'][-1] += output['n_true_positives']
        self.history['semgraph']['fp_tokens'] += output['n_false_positives']
        self.history['semgraph']['fn_tokens'] += output['n_false_negatives']
        self.history['semgraph']['sequences'][-1] += output['n_correct_sequences']
        if 'accessible' in self.history:
          self.history['accessible']['loss'][-1] += output['unlabeled_loss']
          if 'fp_tokens' in self.history['accessible']:
            if isinstance(self.history['accessible']['fp_tokens'], list):
              for i in range(len(self.history['accessible']['fp_tokens'])):
                self.history['accessible']['tokens'][-1][i] += output['n_acc_true_positives'][i]
                self.history['accessible']['fp_tokens'][i] += output['n_acc_false_positives'][i]
                self.history['accessible']['fn_tokens'][i] += output['n_acc_false_negatives'][i]
              #print (self.history['accessible']['fp_tokens'], self.history['accessible']['fn_tokens'])
            else:
              self.history['accessible']['tokens'][-1] += output['n_acc_true_positives']
              self.history['accessible']['fp_tokens'] += output['n_acc_false_positives']
              self.history['accessible']['fn_tokens'] += output['n_acc_false_negatives']
      elif field == 'deptree':
        if self._factored_deptree:
          self.history['deprel']['loss'][-1] += output['label_loss']
          self.history['deprel']['tokens'][-1] += output['n_correct_label_tokens']
          self.history['deprel']['sequences'][-1] += output['n_correct_label_sequences']
        self.history['dephead']['loss'][-1] += output['unlabeled_loss']
        self.history['dephead']['tokens'][-1] += output['n_correct_unlabeled_tokens']
        self.history['dephead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
        self.history['deptree']['loss'][-1] += output['loss']
        self.history['deptree']['tokens'][-1] += output['n_correct_tokens']
        self.history['deptree']['sequences'][-1] += output['n_correct_sequences']
      elif field != 'total':
        self.history[field]['loss'][-1] += output['loss']
        self.history[field]['tokens'][-1] += output['n_correct_tokens']
        self.history[field]['sequences'][-1] += output['n_correct_sequences']
    return
  
  #=============================================================
  def print_recent_history(self, stdscr=None):
    """"""
    
    n_batches = self.history['total']['n_batches']
    n_tokens = self.history['total']['n_tokens']
    n_sequences = self.history['total']['n_sequences']
    total_time = self.history['total']['total_time']
    self.history['total']['n_batches'] = 0
    self.history['total']['n_tokens'] = 0
    self.history['total']['n_sequences'] = 0
    self.history['total']['total_time'] = 0
    
    #-----------------------------------------------------------
    if stdscr is not None:
      stdscr.addstr('{:5}\n'.format(self.dataset.title()), curses.color_pair(1) | curses.A_BOLD)
      stdscr.clrtoeol()
    else:
      print('{:5}\n'.format(self.dataset.title()), end='')
    
    for field, string in self._print_mapping:
      if field in self.history:
        tokens = self.history[field]['tokens'][-1]
        if field in ('semgraph', 'semhead'):
          tp = self.history[field]['tokens'][-1]
          print ('tp:{}, pred:{}, gold:{}'.format(self.history[field]['tokens'][-1],
            self.history[field]['tokens'][-1]+self.history[field]['fp_tokens'],
            self.history[field]['tokens'][-1]+self.history[field]['fn_tokens']))
          self.history[field]['tokens'][-1], precision, recall = [v*100 for v in self.compute_token_F1_PR(field)]
        elif field == 'semrel':
          n_edges = self.history[field]['n_edges']
          self.history[field]['tokens'][-1] *= 100 / n_edges
          self.history[field]['n_edges'] = 0
        else:
          self.history[field]['tokens'][-1] *= 100 / n_tokens
        self.history[field]['loss'][-1] /= n_batches
        self.history[field]['sequences'][-1] *= 100 / n_sequences
        loss = self.history[field]['loss'][-1]
        acc = self.history[field]['tokens'][-1]
        acc_seq = self.history[field]['sequences'][-1]
        if stdscr is not None:
          stdscr.addstr('{:5}'.format(string), curses.color_pair(6) | curses.A_BOLD)
          stdscr.addstr(' | ')
          stdscr.addstr('Loss: {:.2e}'.format(loss), curses.color_pair(3) | curses.A_BOLD)
          stdscr.addstr(' | ')
          stdscr.addstr('Acc: {:5.2f}'.format(acc), curses.color_pair(4) | curses.A_BOLD)
          stdscr.addstr(' | ')
          stdscr.addstr('Seq: {:5.2f}\n'.format(acc_seq), curses.color_pair(4) | curses.A_BOLD)
          stdscr.clrtoeol()
        else:
          print('{:5}'.format(string), end='')
          print(' | ', end='')
          print('Loss: {:.2e}'.format(loss), end='')
          print(' | ', end='')
          print('Acc: {:5.2f}'.format(acc), end='')
          print(' | ', end='')
          print('Seq: {:5.2f}\n'.format(acc_seq), end='')
        if field in ('semgraph', 'semhead'):
          print('{:5}'.format(string), end='')
          print(' | ', end='')
          print('F1: {:5.2f}'.format(acc), end='')
          print(' | ', end='')
          print('P: {:5.2f}'.format(precision), end='')
          print(' | ', end='')
          print('R: {:5.2f}\n'.format(recall), end='')
        for key, value in six.iteritems(self.history[field]):
          if hasattr(value, 'append'):
            value.append(0)
          else:
            self.history[field][key] = 0

    if 'accessible' in self.history:
      field = 'accessible'
      self.history[field]['loss'][-1] /= n_batches
      loss = self.history[field]['loss'][-1]
      self.history[field]['loss'].append(0)
      print('{:5}'.format('ACC'), end='')
      print(' | ', end='')
      print('Loss: {:.2e}\n'.format(loss), end='')
      if 'tokens' in self.history[field]:
        if isinstance(self.history[field]['fp_tokens'], list):
          n_layers = len(self.history[field]['fp_tokens'])
          for i in range(n_layers):
            n_tp = self.history[field]['tokens'][-1][i]
            n_gold = n_tp + self.history[field]['fn_tokens'][i]
            self.history[field]['tokens'][-1][i], precision, recall = [v*100 for v in self.compute_token_F1_PR_index(field, i)]
            acc = self.history[field]['tokens'][-1][i]
            print('{:5}'.format('L-'+str(i)), end='')
            print(' | ', end='')
            print('F1: {:5.2f}'.format(acc), end='')
            print(' | ', end='')
            print('P: {:5.2f}'.format(precision), end='')
            print(' | ', end='')
            print('R: {:5.2f}'.format(recall), end='')
            print(' | ', end='')
            print('gold: {:5.2f}'.format(n_gold), end='')
            print(' | ', end='')
            print('tp: {:5.2f}\n'.format(n_tp), end='')
          self.history[field]['tokens'].append([0]*n_layers)
          self.history[field]['fp_tokens'] = [0]*n_layers
          self.history[field]['fn_tokens'] = [0]*n_layers
        else:
          self.history[field]['tokens'][-1], precision, recall = [v*100 for v in self.compute_token_F1_PR(field)]
          acc = self.history[field]['tokens'][-1]
          print(' | ', end='')
          print('F1: {:5.2f}'.format(acc), end='')
          print(' | ', end='')
          print('P: {:5.2f}'.format(precision), end='')
          print(' | ', end='')
          print('R: {:5.2f}\n'.format(recall), end='')
          self.history[field]['tokens'].append(0)
          self.history[field]['fp_tokens'] = 0
          self.history[field]['fn_tokens'] = 0
      else:
        print ('\n', end='')
    
    
    self.history['speed']['toks/sec'].append(n_tokens / total_time)
    self.history['speed']['seqs/sec'].append(n_sequences / total_time)
    self.history['speed']['bats/sec'].append(n_batches / total_time)
    tps = self.history['speed']['toks/sec'][-1]
    sps = self.history['speed']['seqs/sec'][-1]
    bps = self.history['speed']['bats/sec'][-1]
    if stdscr is not None:
      stdscr.clrtoeol()
      stdscr.addstr('Speed', curses.color_pair(6) | curses.A_BOLD)
      stdscr.addstr(' | ')
      stdscr.addstr('Seqs/sec: {:6.1f}'.format(sps), curses.color_pair(5) | curses.A_BOLD)
      stdscr.addstr(' | ')
      stdscr.addstr('Bats/sec: {:4.2f}\n'.format(bps), curses.color_pair(5) | curses.A_BOLD)
      stdscr.clrtoeol()
      stdscr.addstr('Count', curses.color_pair(6) | curses.A_BOLD)
      stdscr.addstr(' | ')
      stdscr.addstr('Toks: {:6d}'.format(n_tokens), curses.color_pair(7) | curses.A_BOLD)
      stdscr.addstr(' | ')
      stdscr.addstr('Seqs: {:5d}\n'.format(n_sequences), curses.color_pair(7) | curses.A_BOLD)
    else:
      print('Speed', end='')
      print(' | ', end='')
      print('Seqs/sec: {:6.1f}'.format(sps), end='')
      print(' | ', end='')
      print('Bats/sec: {:4.2f}\n'.format(bps), end='')
      print('Count', end='')
      print(' | ', end='')
      print('Toks: {:6d}'.format(n_tokens), end='')
      print(' | ', end='')
      print('Seqs: {:5d}\n'.format(n_sequences), end='')
    filename = os.path.join(self.save_dir, '{}.pkl'.format(self.dataset))
    with open(filename, 'wb') as f:
      pkl.dump(self.history, f, protocol=pkl.HIGHEST_PROTOCOL)
    return
  
  #=============================================================
  @property
  def evals(self):
    return self._evals
  @property
  def accuracies(self):
    return dict(self._accuracies)
  @property
  def probabilities(self):
    return dict(self._probabilities)
  @property
  def loss(self):
    return self._loss
  @property
  def save_dir(self):
    return self._config.getstr(self, 'save_dir')
  @property
  def dataset(self):
    return self._dataset
  @property
  def decoder(self):
    return self._config.getstr(self, 'decoder')
  @property
  def loss_interpolation(self):
    return self._loss_interpolation
  @property
  def losses_by_layer(self):
    return self._losses_by_layer
  @property
  def num_used_layers(self):
    return self._config.getint(self, 'num_used_layers')


#***************************************************************
class TrainOutputs(GraphOutputs):
  _dataset = 'train'
class DevOutputs(GraphOutputs):
  _dataset = 'dev'
