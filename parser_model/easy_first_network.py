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
import time
import os
import pickle as pkl
import curses
import codecs

import numpy as np
import tensorflow as tf

from debug.timer import Timer

from parser_model.structs import conllu_dataset
from parser_model.graph_outputs import GraphOutputs, TrainOutputs, DevOutputs
from parser_model.neural.optimizers import AdamOptimizer, AMSGradOptimizer
from parser_model.neural.optimizers.bert_adam import create_optimizer

from parser_model.base_network import BaseNetwork
from parser_model.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser_model.neural import easy_first_transformer, graph_transformer

#***************************************************************
class EasyFirstNetwork(BaseNetwork):
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
    # set the the first column of token_weights to be 1
    #root_weights = token_weights + (1-nn.greater(tf.range(bucket_size), 0))
    root_weights = token_weights + (nn.equal(tf.range(bucket_size), 1))
    # this is mask for arc/label prediction (shape = [batch_size, bucket_size, bucket_size])
    token_weights3D = tf.expand_dims(token_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
    tokens = {'n_tokens': n_tokens,
              'tokens_per_sequence': tokens_per_sequence,
              'token_weights': token_weights,
              'token_weights3D': token_weights,
              'n_sequences': n_sequences}

    # shape = [batch_size, bucket_size], the null token is 1, all others are 0
    null_mask = nn.equal(self.id_vocab.placeholder, -1)

    if self.n_std_layers > 0:
      config_std = graph_transformer.GraphTransformerConfig(hidden_size=self.hidden_size,
                                                        num_hidden_layers=self.n_std_layers,
                                                        num_attention_heads=self.n_attention_heads,
                                                        intermediate_size=self.intermediate_size,
                                                        hidden_act="gelu",
                                                        hidden_dropout_prob=self.hidden_dropout_prob,
                                                        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                                                        max_position_embeddings=self.max_position_embeddings,
                                                        initializer_range=0.02,
                                                        supervision='none')

      with tf.variable_scope('Transformer-std'):
        # shape = [batch_size, seq_len, seq_len]
        #input_mask_3D = tf.expand_dims(root_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
        transformer = graph_transformer.GraphTransformer(config_std, not reuse, layer, 
                                                          input_mask=token_weights3D,
                                                          accessible_matrices=None)
        # shape = [batch_size, seq_len, hidden_size]
        layer = transformer.get_sequence_output()

    config = easy_first_transformer.EasyFirstTransformerConfig(hidden_size=self.hidden_size,
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
                                                      rm_prev_tp=self.rm_prev_tp,
                                                      num_sup_heads=self.n_supervised_attention_heads,
                                                      n_top_heads=self.n_top_selected_depheads,
                                                      use_biaffine=self.use_biaffine,
                                                      arc_hidden_size=self.arc_hidden_size,
                                                      arc_hidden_add_linear=self.arc_hidden_add_linear,
                                                      arc_hidden_keep_prob=self.arc_hidden_keep_prob,
                                                      rel_hidden_size=self.rel_hidden_size,
                                                      rel_hidden_add_linear=self.rel_hidden_add_linear,
                                                      rel_hidden_keep_prob=self.rel_hidden_keep_prob,
                                                      sample_policy=self.sample_policy,
                                                      share_attention_params=self.share_attention_params)

    output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
    outputs = {}

    with tf.variable_scope('Transformer'):
      # shape = [batch_size, seq_len, seq_len]
      #input_mask_3D = tf.expand_dims(root_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
      unlabeled_targets = output_fields['semhead'].placeholder
      transformer = easy_first_transformer.EasyFirstTransformer(config, not reuse, layer, 
                                                        input_mask=token_weights3D,
                                                        unlabeled_targets=unlabeled_targets,
                                                        null_mask=null_mask)
      if self.concat_all_layers:
        layers = transformer.get_all_encoder_layers()
        # shape = [batch_size, seq_len, hidden_size*n_layers]
        concat_layer = tf.concat(layers, -1)
        # shape = [batch_size, seq_len, hidden_size]
        layer = tf.layers.dense(
            concat_layer,
            self.hidden_size,
            activation=None,
            name="layers2hidden",
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
      else:
        # shape = [batch_size, seq_len, hidden_size]
        layer = transformer.get_sequence_output()

      acc_outputs = transformer.get_outputs()
      losses_by_layer = acc_outputs['acc_loss']
      if not reuse and self.n_steps_change_loss_weight > 0:
        self.loss_weights = tf.placeholder(tf.float32, shape=[self.n_layers], name='loss-weights')
        acc_outputs['acc_loss'] = tf.reduce_sum(self.loss_weights * tf.stack(acc_outputs['acc_loss']))
      else:
        acc_outputs['acc_loss'] = tf.add_n(acc_outputs['acc_loss'])
    
    unlabeled_outputs = {}
    unlabeled_outputs['unlabeled_targets'] = output_fields['semhead'].placeholder
    #unlabeled_outputs['probabilities'] = acc_outputs['probabilities']
    # [batch_size, seq_len, seq_len], for input of rel classifier to calculate the final probability
    # probabilities = label_probabilities * head_probabilities
    unlabeled_outputs['probabilities'] = tf.ones_like(acc_outputs['probabilities'][0][0])
    unlabeled_outputs['unlabeled_loss'] = acc_outputs['acc_loss']
    unlabeled_outputs['loss'] = acc_outputs['acc_loss']
    
    unlabeled_outputs['unlabeled_predictions'] = acc_outputs['predictions']
    unlabeled_outputs['n_unlabeled_true_positives'] = acc_outputs['n_unlabeled_true_positives']
    unlabeled_outputs['n_unlabeled_false_positives'] = acc_outputs['n_unlabeled_false_positives']
    unlabeled_outputs['n_unlabeled_false_negatives'] = acc_outputs['n_unlabeled_false_negatives']
    unlabeled_outputs['n_correct_unlabeled_sequences'] = acc_outputs['n_correct_unlabeled_sequences']
    unlabeled_outputs['predictions'] = acc_outputs['predictions']

    with tf.variable_scope('Classifiers'):
      if 'semrel' in output_fields:
        vocab = output_fields['semrel']
        with tf.variable_scope('Labeled'):
          labeled_outputs = vocab.get_bilinear_classifier(
              layer, unlabeled_outputs,
              token_weights=token_weights3D,
              reuse=reuse)
        outputs['semgraph'] = labeled_outputs
        #if acc_outputs is not None:
          #outputs['semgraph']['loss'] += acc_outputs['acc_loss']
          #for field in acc_outputs:
          #  if field == 'probabilities':
          #    outputs['acc'] = {field : acc_outputs[field]}
          #  else:
          #    outputs['semgraph'][field] = acc_outputs[field]
        self._evals.add('semgraph')
    if self.optimize_by_layer:
      outputs['semgraph']['losses_by_layer'] = losses_by_layer
    outputs['semgraph']['preds_by_layer'] = acc_outputs['preds_by_layer']
    outputs['semgraph']['allowed_heads'] = acc_outputs['allowed_heads']
    outputs['semgraph']['used_heads'] = acc_outputs['used_heads']
    outputs['semgraph']['unlabeled_probabilities'] = acc_outputs['probabilities']
    return outputs, tokens
  
  #=============================================================
  def train(self, load=False, noscreen=False):
    """"""

    trainset = conllu_dataset.CoNLLUTrainset(self.vocabs,
                                             config=self._config, add_null_token=True)
    devset = conllu_dataset.CoNLLUDevset(self.vocabs,
                                         config=self._config, add_null_token=True)
    #testset = conllu_dataset.CoNLLUTestset(self.vocabs, config=self._config)

    factored_deptree = None
    factored_semgraph = None
    for vocab in self.output_vocabs:
      if vocab.field == 'deprel':
        factored_deptree = vocab.factorized
      elif vocab.field == 'semrel':
        factored_semgraph = vocab.factorized

    input_network_outputs = {}
    input_network_savers = []
    input_network_paths = []

    for input_network in self.input_networks:
      with tf.variable_scope(input_network.classname, reuse=False):
        input_network_outputs[input_network.classname] = input_network.build_graph(reuse=True)[0]

      network_variables = set(tf.global_variables(scope=input_network.classname))
      non_save_variables = set(tf.get_collection('non_save_variables'))
      network_save_variables = network_variables - non_save_variables
      saver = tf.train.Saver(list(network_save_variables))
      input_network_savers.append(saver)
      input_network_paths.append(self._config(self, input_network.classname+'_dir'))

    with tf.variable_scope(self.classname, reuse=False):
      train_graph = self.build_graph(input_network_outputs=input_network_outputs, reuse=False)
      train_outputs = TrainOutputs(*train_graph, load=load, evals=self._evals, \
                                   factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)

    with tf.variable_scope(self.classname, reuse=True):
      dev_graph = self.build_graph(input_network_outputs=input_network_outputs, reuse=True)
      dev_outputs = DevOutputs(*dev_graph, load=load, evals=self._evals, \
                               factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)

    regularization_loss = self.l2_reg * tf.losses.get_regularization_loss() if self.l2_reg else 0

    update_step = tf.assign_add(self.global_step, 1)

    if train_outputs.losses_by_layer:
      print ("optimize by layer")
      adam_ops = []
      adam_optimizers = []
      for n_layer, loss in enumerate(train_outputs.losses_by_layer):
        if self.use_bert_adam:
          adam_op = create_optimizer(loss=loss + regularization_loss,
                                     init_lr=float(self._config._sections['Optimizer']['learning_rate']),
                                     bert_lr=float(self._config._sections['Optimizer']['bert_learning_rate']),
                                     num_train_steps=self.max_steps,
                                     num_warmup_steps=self.warm_up_steps,
                                     use_tpu=False)
        else:
          with tf.variable_scope('layer-{}'.format(n_layer)):
            adam = AdamOptimizer(config=self._config)
            adam_op = adam.minimize(loss + regularization_loss, \
                                  variables=tf.trainable_variables(scope=self.classname)) # returns the current step
            adam_optimizers.append(adam)
        # not save the op for last layer, leave it with accuracies
        if n_layer < len(train_outputs.losses_by_layer)-1:
          adam_ops.append(adam_op)

      adam_train_tensors = [adam_op, train_outputs.accuracies]

      if self.switch_optimizers:
        amsgrad_ops = []
        for n_layer, loss in enumerate(train_outputs.losses_by_layer):
          with tf.variable_scope('layer-{}'.format(n_layer)):
            amsgrad = AMSGradOptimizer.from_optimizer(adam_optimizers[n_layer])
            amsgrad_op = amsgrad.minimize(train_outputs.loss + regularization_loss, \
                                        variables=tf.trainable_variables(scope=self.classname)) # returns the current step
          if n_layer < len(train_outputs.losses_by_layer)-1:
            amsgrad_ops.append(amsgrad_op)
        amsgrad_train_tensors = [amsgrad_op, train_outputs.accuracies]

      dev_tensors = dev_outputs.accuracies

      # I think this needs to come after the optimizers
      if self.save_model_after_improvement or self.save_model_after_training:
        all_variables = set(tf.global_variables(scope=self.classname))
        non_save_variables = set(tf.get_collection('non_save_variables'))
        save_variables = all_variables - non_save_variables
        saver = tf.train.Saver(list(save_variables), max_to_keep=1)

      screen_output = []
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.allow_soft_placement = True
      with tf.Session(config=config) as sess:
        for saver, path in zip(input_network_savers, input_network_paths):
          saver.restore(sess, path)

        feed_dict = {}
        if self.use_elmo:
          feed_dict[self.elmo_vocabs[0].embed_placeholder] = self.elmo_vocabs[0].embeddings
        if self.use_pretrained:
          feed_dict[self.pretrained_vocabs[0].embed_placeholder] = self.pretrained_vocabs[0].embeddings
        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        for vocab in self.input_vocabs:
          if 'BERT' in vocab.classname:
            with Timer('Restoring BERT pretrained ckpt'):
              bert_scope_name = self.classname + '/Embeddings/' + vocab.classname
              bert_variables = [v for v in tf.global_variables(scope=bert_scope_name) if 'bert' in v.name and 'adam' not in v.name]
              bert_saver = tf.train.Saver({v.name[len(bert_scope_name) + 1:].rsplit(':', maxsplit=1)[0]: v for v in bert_variables})
              bert_saver.restore(sess, vocab.pretrained_ckpt)

        #---
        #os.makedirs(os.path.join(self.save_dir, 'profile'))
        #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        #---
        if not noscreen:
          print ("Removed")
          exit(1)

        self.feed_loss_weight = False
        if hasattr(self, 'n_steps_change_loss_weight') and self.n_steps_change_loss_weight > 0:
          #print (self.n_steps_change_loss_weight)
          self.feed_loss_weight = True
          self.main_loss_layer_id = 0
          assert hasattr(self, 'n_layers')
          assert hasattr(self, 'main_loss_weight')
          assert hasattr(self, 'aux_loss_weight')
          self._loss_weights = [self.aux_loss_weight] * self.n_layers
          self._loss_weights[self.main_loss_layer_id] = self.main_loss_weight
        current_optimizer = 'Adam'
        train_ops = adam_ops
        train_tensors = adam_train_tensors
        current_step = 0
        print('\t', end='')
        print('{}\n'.format(self.save_dir), end='')
        print('\t', end='')
        print('GPU: {}\n'.format(self.cuda_visible_devices), end='')
        try:
          current_epoch = 0
          best_accuracy = 0
          current_accuracy = 0
          steps_since_best = 0
          while (not self.max_steps or current_step < self.max_steps) and \
                (not self.max_steps_without_improvement or steps_since_best < self.max_steps_without_improvement) and \
                (not self.n_passes or current_epoch < len(trainset.conllu_files)*self.n_passes):
            if steps_since_best >= 1 and self.switch_optimizers and current_optimizer != 'AMSGrad':
              train_ops = amsgrad_ops
              train_tensors = amsgrad_train_tensors
              current_optimizer = 'AMSGrad'
              print('\t', end='')
              print('Current optimizer: {}\n'.format(current_optimizer), end='')
            for batch in trainset.batch_iterator(shuffle=True):
              train_outputs.restart_timer()
              start_time = time.time()
              feed_dict = trainset.set_placeholders(batch)
              if self.feed_loss_weight:
                # add loss weight to feed_dict
                feed_dict[self.loss_weights] = self._loss_weights
                # if reach change point, change loss weights
                if (current_step + 1) % self.n_steps_change_loss_weight == 0:
                  self.main_loss_layer_id = (self.main_loss_layer_id + 1) % self.n_layers
                  self._loss_weights = [self.aux_loss_weight] * self.n_layers
                  self._loss_weights[self.main_loss_layer_id] = self.main_loss_weight
                  print ('Change loss weights to {}\n'.format(self._loss_weights))
              #---
              """
              if current_step < 0:
                # update the first to n-1 layer
                for n_layer, train_op in enumerate(train_ops):
                  #print ('Updating loss of layer-{}'.format(n_layer))
                  _ = sess.run(train_op, feed_dict=feed_dict, options=options)
                # update the last layer and get the scores
                _, train_scores = sess.run(train_tensors, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(os.path.join(self.save_dir, 'profile', 'timeline_step_%d.json' % current_step), 'w') as f:
                  f.write(chrome_trace)
              else:
              """
              # update the first to n-1 layer
              for n_layer, train_op in enumerate(train_ops):
                #print ('Updating loss of layer-{}'.format(n_layer))
                #_ = sess.run(train_op, feed_dict=feed_dict, options=options)
                _ = sess.run(train_op, feed_dict=feed_dict)
                # update the last layer and get the scores
                _, train_scores = sess.run(train_tensors, feed_dict=feed_dict)
              #---
              train_outputs.update_history(train_scores)
              current_step += 1
              if current_step % self.print_every == 0:
                for batch in devset.batch_iterator(shuffle=False):
                  dev_outputs.restart_timer()
                  feed_dict = devset.set_placeholders(batch)
                  dev_scores = sess.run(dev_tensors, feed_dict=feed_dict)
                  dev_outputs.update_history(dev_scores, show=True)
                current_accuracy *= .5
                current_accuracy += .5*dev_outputs.get_current_accuracy()
                if current_accuracy >= best_accuracy:
                  steps_since_best = 0
                  best_accuracy = current_accuracy
                  if self.save_model_after_improvement:
                    saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
                else:
                  steps_since_best += self.print_every
                current_epoch = sess.run(self.global_step)
                print('\t', end='')
                print('Epoch: {:3d}'.format(int(current_epoch)), end='')
                print(' | ', end='')
                print('Step: {:5d}\n'.format(int(current_step)), end='')
                print('\t', end='')
                print('Moving acc: {:5.2f}'.format(current_accuracy), end='')
                print(' | ', end='')
                print('Best moving acc: {:5.2f}\n'.format(best_accuracy), end='')
                print('\t', end='')
                print('Steps since improvement: {:4d}\n'.format(int(steps_since_best)), end='')
                train_outputs.print_recent_history()
                dev_outputs.print_recent_history()
            current_epoch = sess.run(self.global_step)
            sess.run(update_step)
            trainset.load_next()
          with open(os.path.join(self.save_dir, 'SUCCESS'), 'w') as f:
            pass
        except KeyboardInterrupt:
          pass
        if self.save_model_after_training:
          saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
    # optimize together
    else:
      print ("optimize together")
      if self.use_bert_adam:
        adam_op = create_optimizer(loss=train_outputs.loss + regularization_loss,
                                   init_lr=float(self._config._sections['Optimizer']['learning_rate']),
                                   bert_lr=float(self._config._sections['Optimizer']['bert_learning_rate']),
                                   num_train_steps=self.max_steps,
                                   num_warmup_steps=self.warm_up_steps,
                                   use_tpu=False)
      else:
        adam = AdamOptimizer(config=self._config)
        adam_op = adam.minimize(train_outputs.loss + regularization_loss, \
                                variables=tf.trainable_variables(scope=self.classname)) # returns the current step
      adam_train_tensors = [adam_op, train_outputs.accuracies]

      if self.switch_optimizers:
        amsgrad = AMSGradOptimizer.from_optimizer(adam)
        amsgrad_op = amsgrad.minimize(train_outputs.loss + regularization_loss, \
                                      variables=tf.trainable_variables(scope=self.classname)) # returns the current step
        amsgrad_train_tensors = [amsgrad_op, train_outputs.accuracies]

      dev_tensors = dev_outputs.accuracies

      # I think this needs to come after the optimizers
      if self.save_model_after_improvement or self.save_model_after_training:
        all_variables = set(tf.global_variables(scope=self.classname))
        non_save_variables = set(tf.get_collection('non_save_variables'))
        save_variables = all_variables - non_save_variables
        saver = tf.train.Saver(list(save_variables), max_to_keep=1)

      screen_output = []
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.allow_soft_placement = True
      with tf.Session(config=config) as sess:
        for saver, path in zip(input_network_savers, input_network_paths):
          saver.restore(sess, path)

        feed_dict = {}
        if self.use_elmo:
          feed_dict[self.elmo_vocabs[0].embed_placeholder] = self.elmo_vocabs[0].embeddings
        if self.use_pretrained:
          feed_dict[self.pretrained_vocabs[0].embed_placeholder] = self.pretrained_vocabs[0].embeddings
        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        for vocab in self.input_vocabs:
          if 'BERT' in vocab.classname:
            with Timer('Restoring BERT pretrained ckpt'):
              bert_scope_name = self.classname + '/Embeddings/' + vocab.classname
              bert_variables = [v for v in tf.global_variables(scope=bert_scope_name) if 'bert' in v.name and 'adam' not in v.name]
              bert_saver = tf.train.Saver({v.name[len(bert_scope_name) + 1:].rsplit(':', maxsplit=1)[0]: v for v in bert_variables})
              bert_saver.restore(sess, vocab.pretrained_ckpt)

        #---
        #os.makedirs(os.path.join(self.save_dir, 'profile'))
        #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        #---
        if not noscreen:
          print ("Removed")
          exit(1)

        self.feed_loss_weight = False
        if hasattr(self, 'n_steps_change_loss_weight') and self.n_steps_change_loss_weight > 0:
          #print (self.n_steps_change_loss_weight)
          self.feed_loss_weight = True
          self.main_loss_layer_id = 0
          assert hasattr(self, 'n_layers')
          assert hasattr(self, 'main_loss_weight')
          assert hasattr(self, 'aux_loss_weight')
          self._loss_weights = [self.aux_loss_weight] * self.n_layers
          self._loss_weights[self.main_loss_layer_id] = self.main_loss_weight
        current_optimizer = 'Adam'
        train_tensors = adam_train_tensors
        current_step = 0
        print('\t', end='')
        print('{}\n'.format(self.save_dir), end='')
        print('\t', end='')
        print('GPU: {}\n'.format(self.cuda_visible_devices), end='')
        try:
          current_epoch = 0
          best_accuracy = 0
          current_accuracy = 0
          steps_since_best = 0
          while (not self.max_steps or current_step < self.max_steps) and \
                (not self.max_steps_without_improvement or steps_since_best < self.max_steps_without_improvement) and \
                (not self.n_passes or current_epoch < len(trainset.conllu_files)*self.n_passes):
            if steps_since_best >= 1 and self.switch_optimizers and current_optimizer != 'AMSGrad':
              train_tensors = amsgrad_train_tensors
              current_optimizer = 'AMSGrad'
              print('\t', end='')
              print('Current optimizer: {}\n'.format(current_optimizer), end='')
            for batch in trainset.batch_iterator(shuffle=True):
              train_outputs.restart_timer()
              start_time = time.time()
              feed_dict = trainset.set_placeholders(batch)
              if self.feed_loss_weight:
                # add loss weight to feed_dict
                feed_dict[self.loss_weights] = self._loss_weights
                # if reach change point, change loss weights
                if (current_step + 1) % self.n_steps_change_loss_weight == 0:
                  self.main_loss_layer_id = (self.main_loss_layer_id + 1) % self.n_layers
                  self._loss_weights = [self.aux_loss_weight] * self.n_layers
                  self._loss_weights[self.main_loss_layer_id] = self.main_loss_weight
                  print ('Change loss weights to {}\n'.format(self._loss_weights))
              #---
              _, train_scores = sess.run(train_tensors, feed_dict=feed_dict)
              #---
              train_outputs.update_history(train_scores)
              current_step += 1
              if current_step % self.print_every == 0:
                for batch in devset.batch_iterator(shuffle=False):
                  dev_outputs.restart_timer()
                  feed_dict = devset.set_placeholders(batch)
                  dev_scores = sess.run(dev_tensors, feed_dict=feed_dict)
                  dev_outputs.update_history(dev_scores, show=True)
                current_accuracy *= .5
                current_accuracy += .5*dev_outputs.get_current_accuracy()
                if current_accuracy >= best_accuracy:
                  steps_since_best = 0
                  best_accuracy = current_accuracy
                  if self.save_model_after_improvement:
                    saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
                else:
                  steps_since_best += self.print_every
                current_epoch = sess.run(self.global_step)
                print('\t', end='')
                print('Epoch: {:3d}'.format(int(current_epoch)), end='')
                print(' | ', end='')
                print('Step: {:5d}\n'.format(int(current_step)), end='')
                print('\t', end='')
                print('Moving acc: {:5.2f}'.format(current_accuracy), end='')
                print(' | ', end='')
                print('Best moving acc: {:5.2f}\n'.format(best_accuracy), end='')
                print('\t', end='')
                print('Steps since improvement: {:4d}\n'.format(int(steps_since_best)), end='')
                train_outputs.print_recent_history()
                dev_outputs.print_recent_history()
            current_epoch = sess.run(self.global_step)
            sess.run(update_step)
            trainset.load_next()
          with open(os.path.join(self.save_dir, 'SUCCESS'), 'w') as f:
            pass
        except KeyboardInterrupt:
          pass
        if self.save_model_after_training:
          saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
    return

  #=============================================================
  def parse(self, conllu_files, output_dir=None, output_filename=None, augment_layers=None):
    """"""

    with Timer('Building dataset'):
      parseset = conllu_dataset.CoNLLUDataset(conllu_files, self.vocabs,
                                              config=self._config, add_null_token=True)

    if output_filename:
      assert len(conllu_files) == 1, "output_filename can only be specified for one input file"
    factored_deptree = None
    factored_semgraph = None
    for vocab in self.output_vocabs:
      if vocab.field == 'deprel':
        factored_deptree = vocab.factorized
      elif vocab.field == 'semrel':
        factored_semgraph = vocab.factorized
    with Timer('Building TF'):
      with tf.variable_scope(self.classname, reuse=False):
        parse_graph = self.build_graph(reuse=True)
        parse_outputs = DevOutputs(*parse_graph, load=False, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)
      parse_tensors = parse_outputs.accuracies
      all_variables = set(tf.global_variables())
      non_save_variables = set(tf.get_collection('non_save_variables'))
      save_variables = all_variables - non_save_variables
      saver = tf.train.Saver(list(save_variables), max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
      with Timer('Initializing non_save variables'):
        #print(list(non_save_variables))
        feed_dict = {}
      if self.use_elmo:
        feed_dict[self.elmo_vocabs[0].embed_placeholder] = self.elmo_vocabs[0].embeddings
      if self.use_pretrained:
        feed_dict[self.pretrained_vocabs[0].embed_placeholder] = self.pretrained_vocabs[0].embeddings
      sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
      with Timer('Restoring save variables'):
        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
      if len(conllu_files) == 1 or output_filename is not None:
        with Timer('Parsing file'):
          if not self.other_save_dirs:
            self.parse_file(parseset, parse_outputs, sess, output_dir=output_dir, 
                            output_filename=output_filename, augment_layers=augment_layers)
          else:
            self.parse_file_ensemble(parseset, parse_outputs, sess, saver, output_dir=output_dir, output_filename=output_filename)
      else:
        with Timer('Parsing files'):
          self.parse_files(parseset, parse_outputs, sess, output_dir=output_dir)
    return

  #=============================================================
  def parse_file(self, dataset, graph_outputs, sess, output_dir=None, output_filename=None,
                  print_time=True, augment_layers=None):
    """"""

    probability_tensors = graph_outputs.probabilities
    input_filename = dataset.conllu_files[0]
    graph_outputs.restart_timer()
    for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
      with Timer('Parsing batch %d' % i):
        tokens, lengths = dataset.get_tokens(indices)
        # remove the null token
        for field in tokens:
          for i in range(len(tokens[field])):
            tokens[field][i] = tokens[field][i][1:]
        lengths = lengths - 1
        feed_dict = dataset.set_placeholders(indices)
        probabilities = sess.run(probability_tensors, feed_dict=feed_dict)
        predictions = graph_outputs.probs_to_preds(probabilities, lengths, augment_layers=augment_layers)
        tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.output_vocabs})
        graph_outputs.cache_predictions(tokens, indices)

    with Timer('Dumping predictions'):
      if output_dir is None and output_filename is None:
        graph_outputs.print_current_predictions()
      else:
        input_dir, input_filename = os.path.split(input_filename)
        if output_dir is None:
          output_dir = os.path.join(self.save_dir, 'parsed', input_dir)
        elif output_filename is None:
          output_filename = input_filename
          output_filename = os.path.join(output_dir, output_filename)
        
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        #output_filename = os.path.join(output_dir, output_filename)
        with codecs.open(output_filename, 'w', encoding='utf-8') as f:
          graph_outputs.dump_current_predictions(f)
    if print_time:
      print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - graph_outputs.time))
    return


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
  def n_std_layers(self):
    return self._config.getint(self, 'n_std_layers')
  @property
  def rm_prev_tp(self):
    return self._config.getboolean(self, 'rm_prev_tp')
  @property
  def concat_all_layers(self):
    return self._config.getboolean(self, 'concat_all_layers')
  @property
  def n_steps_change_loss_weight(self):
    return self._config.getint(self, 'n_steps_change_loss_weight')
  @property
  def main_loss_weight(self):
    return self._config.getfloat(self, 'main_loss_weight')
  @property
  def aux_loss_weight(self):
    return self._config.getfloat(self, 'aux_loss_weight')
  @property
  def n_supervised_attention_heads(self):
    return self._config.getint(self, 'n_supervised_attention_heads')
  @property
  def n_top_selected_depheads(self):
    return self._config.getint(self, 'n_top_selected_depheads')
  @property
  def optimize_by_layer(self):
    return self._config.getboolean(self, 'optimize_by_layer')
  @property
  def use_biaffine(self):
    return self._config.getboolean(self, 'use_biaffine')
  @property
  def arc_hidden_size(self):
    return self._config.getint(self, 'arc_hidden_size')
  @property
  def arc_hidden_add_linear(self):
    return self._config.getboolean(self, 'arc_hidden_add_linear')
  @property
  def arc_hidden_keep_prob(self):
    return self._config.getfloat(self, 'arc_hidden_keep_prob')
  @property
  def rel_hidden_size(self):
    return self._config.getint(self, 'rel_hidden_size')
  @property
  def rel_hidden_add_linear(self):
    return self._config.getboolean(self, 'rel_hidden_add_linear')
  @property
  def rel_hidden_keep_prob(self):
    return self._config.getfloat(self, 'rel_hidden_keep_prob')
  @property
  def sample_policy(self):
    return self._config.getstr(self, 'sample_policy')
  @property
  def share_attention_params(self):
    return self._config.getboolean(self, 'share_attention_params')