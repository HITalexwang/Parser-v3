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

import re
import time
import os
import pickle as pkl
import curses
import codecs

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from debug.timer import Timer

from parser.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser.graph_outputs import GraphOutputs, TrainOutputs, DevOutputs, AuxOutputs
from parser.structs import conllu_dataset
from parser.structs import vocabs
from parser.neural.optimizers import AdamOptimizer, AMSGradOptimizer

#***************************************************************
class BaseNetwork(object):
  """"""

  _evals = set()

  #=============================================================
  def __init__(self, input_networks=set(), config=None):
    """"""
    
    with Timer('Initializing the network (including pretrained vocab)'):
      self._config = config
      #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
      self._input_networks = input_networks
      input_network_classes = set(input_network.classname for input_network in self._input_networks)
      assert input_network_classes == set(self.input_network_classes), 'Not all input networks were passed in to {}'.format(self.classname)

      extant_vocabs = {}
      for input_network in self.input_networks:
        for vocab in input_network.vocabs:
          if vocab.classname in extant_vocabs:
            assert vocab is extant_vocabs[vocab.classname], "Two input networks have different instances of {}".format(vocab.classname)
          else:
            extant_vocabs[vocab.classname] = vocab

      if 'IDIndexVocab' in extant_vocabs:
        self._id_vocab = extant_vocabs['IDIndexVocab']
      else:
        self._id_vocab = vocabs.IDIndexVocab(config=config)
        extant_vocabs['IDIndexVocab'] = self._id_vocab

      aux_conllus = self.aux_conllus or None
      self._input_vocabs = []
      for input_vocab_classname in self.input_vocab_classes:
        if input_vocab_classname in extant_vocabs:
          self._input_vocabs.append(extant_vocabs[input_vocab_classname])
        else:
          VocabClass = getattr(vocabs, input_vocab_classname)
          vocab = VocabClass(config=config)
          vocab.load() or vocab.count(self.train_conllus, aux_conllus=aux_conllus)
          self._input_vocabs.append(vocab)
          extant_vocabs[input_vocab_classname] = vocab

      self._output_vocabs = []
      for output_vocab_classname in self.output_vocab_classes:
        if output_vocab_classname in extant_vocabs:
          self._output_vocabs.append(extant_vocabs[output_vocab_classname])
        else:
          VocabClass = getattr(vocabs, output_vocab_classname)
          vocab = VocabClass(config=config)
          if 'Semrel' in output_vocab_classname and not self.aux_label:
            vocab.load() or vocab.count(self.train_conllus, aux_conllus=None)
          else:
            vocab.load() or vocab.count(self.train_conllus, aux_conllus=aux_conllus)
          self._output_vocabs.append(vocab)
          extant_vocabs[output_vocab_classname] = vocab

      self._throughput_vocabs = []
      for throughput_vocab_classname in self.throughput_vocab_classes:
        if throughput_vocab_classname in extant_vocabs:
          self._throughput_vocabs.append(extant_vocabs[throughput_vocab_classname])
        else:
          VocabClass = getattr(vocabs, throughput_vocab_classname)
          vocab = VocabClass(config=config)
          vocab.load() or vocab.count(self.train_conllus, aux_conllus=aux_conllus)
          self._throughput_vocabs.append(vocab)
          extant_vocabs[throughput_vocab_classname] = vocab

      with tf.variable_scope(self.classname, reuse=False):
        self.global_step = tf.Variable(0., trainable=False, name='Global_step')
      self._vocabs = set(extant_vocabs.values())
    return

  #=============================================================
  def get_mix_rate_upbounds(self):
    mix_rates = [float(rate) for rate in self.mix_rates]
    sum_rate = sum(mix_rates)
    mix_rates = [rate/sum_rate for rate in mix_rates]
    self._mix_rate_upbounds = [mix_rates[0]]
    for i in range(1, len(mix_rates)):
      self._mix_rate_upbounds.append(self._mix_rate_upbounds[i-1]+mix_rates[i])
    return

  #=============================================================
  def next_batch(self, iters, sets):
    """"""

    update = False
    # no aux set
    if len(iters) == 1:
      batch = next(iters[0], None)
      if batch is None:
        print ("### Reload trainset batches ###")
        update = True
        sets[0].load_next()
        iters[0] = sets[0].batch_iterator(shuffle=True)
        batch = next(iters[0], None)
      feed_dict = sets[0].set_placeholders(batch)
      return feed_dict, 0, update
    # choose a set where the next batch come from
    else:
      assert len(self.mix_rate_upbounds) == len(iters)
      rand = np.random.rand()
      for i in range(len(self.mix_rate_upbounds)):
        if rand <= self.mix_rate_upbounds[i]: break
      #print (self.mix_rate_upbounds, rand, i)
      batch = next(iters[i], None)
      if batch is None:
        print ("### Reload set-{} batches ###".format(i))
        iters[i] = sets[i].batch_iterator(shuffle=True)
        batch = next(iters[i], None)
        if i == 0:
          update = True
          sets[0].load_next()
      feed_dict = sets[i].set_placeholders(batch)
      return feed_dict, i, update

  #=============================================================
  def train(self, load=False, noscreen=False):
    """"""

    trainset = conllu_dataset.CoNLLUTrainset(self.vocabs,
                                             config=self._config)
    devset = conllu_dataset.CoNLLUDevset(self.vocabs,
                                         config=self._config)

    #testset = conllu_dataset.CoNLLUTestset(self.vocabs, config=self._config)
    use_aux = True if self.aux_conllus else False
    auxsets = []
    if use_aux:
      auxsets = [conllu_dataset.CoNLLUAuxset([aux_conllu], self.vocabs, config=self._config) for aux_conllu in self.aux_conllus]
    print ("### Using {} Auxiliary Set(s) ###".format(len(auxsets)))

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
      #with tf.device('/gpu:0'):
      train_graph = self.build_graph(input_network_outputs=input_network_outputs, reuse=False, n_aux=len(auxsets))
      aux_outputs = None
      if use_aux:
        aux_graphs = []
        aux_outputs = []
        for i in range(len(auxsets)):
          aux_graphs.append([{'auxgraph':train_graph[0].pop('auxgraph-%d'%i)}])
          aux_graphs[-1].append(train_graph[1])
          aux_outputs.append(AuxOutputs(*aux_graphs[i], load=load, evals=self._evals, factored_deptree=False, 
                                          factored_semgraph=False, config=self._config, dataset='aux-%d'%i))
      """
      if use_aux and 'auxhead' in train_graph[0]:
        aux_graph = [{'auxhead':train_graph[0].pop('auxhead')}]
        aux_graph.append(train_graph[1])
        aux_outputs = AuxOutputs(*aux_graph, load=load, evals=self._evals, factored_deptree=False, factored_semgraph=False, config=self._config)
      elif 'auxhead' in  train_graph[0]:
        train_graph[0].pop('auxhead')
      """
      train_outputs = TrainOutputs(*train_graph, load=load, evals=self._evals, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)
    with tf.variable_scope(self.classname, reuse=True):
      #with tf.device('/gpu:0'):
      dev_graph = self.build_graph(input_network_outputs=input_network_outputs, reuse=True)
      dev_outputs = DevOutputs(*dev_graph, load=load, evals=self._evals, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self._config)
    regularization_loss = self.l2_reg * tf.losses.get_regularization_loss() if self.l2_reg else 0

    update_step = tf.assign_add(self.global_step, 1)
    adam = AdamOptimizer(config=self._config)
    adam_op = adam.minimize(train_outputs.loss + regularization_loss, variables=tf.trainable_variables(scope=self.classname)) # returns the current step
    adam_train_tensors = [adam_op, train_outputs.accuracies]
    amsgrad = AMSGradOptimizer.from_optimizer(adam)
    amsgrad_op = amsgrad.minimize(train_outputs.loss + regularization_loss, variables=tf.trainable_variables(scope=self.classname)) # returns the current step
    amsgrad_train_tensors = [amsgrad_op, train_outputs.accuracies]
    if use_aux and aux_outputs is not None:
      aux_adam_train_tensors = []
      aux_amsgrad_train_tensors = []
      for i in range(len(aux_outputs)):
        aux_adam_op = adam.minimize(aux_outputs[i].loss, variables=tf.trainable_variables(scope=self.classname))
        aux_adam_train_tensors.append([aux_adam_op, aux_outputs[i].accuracies])
        aux_amsgrad_op = amsgrad.minimize(aux_outputs[i].loss, variables=tf.trainable_variables(scope=self.classname))
        aux_amsgrad_train_tensors.append([aux_amsgrad_op, aux_outputs[i].accuracies])
    dev_tensors = dev_outputs.accuracies
    # I think this needs to come after the optimizers
    if self.save_model_after_improvement or self.save_model_after_training:
      all_variables = set(tf.global_variables(scope=self.classname))
      non_save_variables = set(tf.get_collection('non_save_variables'))
      save_variables = all_variables - non_save_variables
      saver = tf.train.Saver(list(save_variables), max_to_keep=1)

    screen_output = []
    gpus = self.cuda_visible_devices.strip().split(',') if self.cuda_visible_devices else []
    config = tf.ConfigProto()
    if self.cpu_num > 0:
      config.device_count['CPU'] = self.cpu_num
    # These 2 should be used to control threads while using only CPU
    if self.intra_threads > 0:
      print ("Intra threads:",self.intra_threads)
      config.intra_op_parallelism_threads=self.intra_threads
    if self.inter_threads > 0:
      print ("Inter threads:",self.inter_threads)
      config.inter_op_parallelism_threads=self.inter_threads
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
      for saver, path in zip(input_network_savers, input_network_paths):
        saver.restore(sess, path)
      sess.run(tf.global_variables_initializer())
      #---
      os.makedirs(os.path.join(self.save_dir, 'profile'))
      options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      #---
      if not noscreen:
        print ("NO Screen Version is removed for simplicity!")
        exit(1)
      else:
        current_optimizer = 'Adam'
        iters = [trainset.batch_iterator(shuffle=True)]
        sets = [trainset]
        outputs = [train_outputs]
        tensors = [adam_train_tensors]
        trains = [0]
        if use_aux:
          trains += [0] * len(auxsets)
          self.get_mix_rate_upbounds()
          iters += [auxset.batch_iterator(shuffle=True) for auxset in auxsets]
          sets += auxsets
          outputs += aux_outputs
          tensors += aux_adam_train_tensors
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
              tensors[0] = amsgrad_train_tensors
              if use_aux:
                for i in range(1, len(tensors)):
                  tensors[i] = aux_amsgrad_train_tensors[i-1]
              current_optimizer = 'AMSGrad'
              print('\t', end='')
              print('Current optimizer: {}\n'.format(current_optimizer), end='')

            feed_dict, task_id, update = self.next_batch(iters, sets)
            trains[task_id] += 1
            if update:
              sess.run(update_step)
            outputs[task_id].restart_timer()
            start_time = time.time()

            with tf.device('/gpu:0'):
              if current_step < 1:
                _, scores = sess.run(tensors[task_id], feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(os.path.join(self.save_dir, 'profile', 'timeline_step_%d.json' % current_step), 'w') as f:
                  f.write(chrome_trace)
              else:
                _, scores = sess.run(tensors[task_id], feed_dict=feed_dict)
              outputs[task_id].update_history(scores)

            """
            # old version
            for batch in trainset.batch_iterator(shuffle=True):
              #print ("### Train on one batch of trainset ###")

              train_outputs.restart_timer()
              start_time = time.time()
              feed_dict = trainset.set_placeholders(batch)
              #---
              with tf.device('/gpu:0'):
                if current_step < 10:
                  _, train_scores = sess.run(train_tensors, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                  chrome_trace = fetched_timeline.generate_chrome_trace_format()
                  with open(os.path.join(self.save_dir, 'profile', 'timeline_step_%d.json' % current_step), 'w') as f:
                    f.write(chrome_trace)
                else:
                  _, train_scores = sess.run(train_tensors, feed_dict=feed_dict)
                #---
                train_outputs.update_history(train_scores)

                # run a auxiliary set batch
                if use_aux:
                  aux_batch = next(aux_iter, None)
                  if aux_batch is None:
                    #print ("### Reload auxset batches ###")
                    aux_iter = auxset.batch_iterator(shuffle=True)
                    aux_batch = next(aux_iter, None)
                  #print ("### Train on one batch of auxset ###")
                  aux_outputs.restart_timer()
                  feed_dict = auxset.set_placeholders(aux_batch)
                  _, aux_scores = sess.run(aux_tensors, feed_dict=feed_dict)
                  aux_outputs.update_history(aux_scores)
              """

            current_step += 1
            if current_step % self.print_every == 0:
              for batch in devset.batch_iterator(shuffle=False):
                dev_outputs.restart_timer()
                feed_dict = devset.set_placeholders(batch)
                dev_scores = sess.run(dev_tensors, feed_dict=feed_dict)
                dev_outputs.update_history(dev_scores)
              current_accuracy *= .5
              current_accuracy += .5*dev_outputs.get_current_accuracy()
              if current_accuracy >= best_accuracy:
                steps_since_best = 0
                best_accuracy = current_accuracy
                if self.save_model_after_improvement:
                  saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
                if self.parse_devset:
                  self.parse_files(devset, dev_outputs, sess, print_time=False)
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
              if use_aux:
                for i in range(len(aux_outputs)):
                  aux_outputs[i].print_recent_history()
              dev_outputs.print_recent_history()
              print ("Batch splits: {}".format(":".join([str(b) for b in trains])))
              for i in range(len(trains)):
                trains[i] = 0
            
            current_epoch = sess.run(self.global_step)
            #sess.run(update_step)
            #trainset.load_next()
          with open(os.path.join(self.save_dir, 'SUCCESS'), 'w') as f:
            pass
        except KeyboardInterrupt:
          pass
      if self.save_model_after_training:
        saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
    return

  #=============================================================
  def parse(self, conllu_files, output_dir=None, output_filename=None):
    """"""

    with Timer('Building dataset'):
      parseset = conllu_dataset.CoNLLUDataset(conllu_files, self.vocabs,
                                              config=self._config)

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
        if 'auxhead' in parse_graph[0]:
          parse_graph[0].pop('auxhead')
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
        print(list(non_save_variables))
        sess.run(tf.variables_initializer(list(non_save_variables)))
      with Timer('Restoring save variables'):
        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
      if len(conllu_files) == 1 or output_filename is not None:
        with Timer('Parsing file'):
          if not self.other_save_dirs:
            self.parse_file(parseset, parse_outputs, sess, output_dir=output_dir, output_filename=output_filename)
          else:
            self.parse_file_ensemble(parseset, parse_outputs, sess, saver, output_dir=output_dir, output_filename=output_filename)
      else:
        with Timer('Parsing files'):
          self.parse_files(parseset, parse_outputs, sess, output_dir=output_dir)
    return

  #=============================================================
  def parse_file(self, dataset, graph_outputs, sess, output_dir=None, output_filename=None, print_time=True):
    """"""

    probability_tensors = graph_outputs.probabilities
    score_tensors = graph_outputs.accuracies
    input_filename = dataset.conllu_files[0]
    #graph_outputs.restart_timer()
    start_time = time.time()
    for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
      with Timer('Parsing batch %d' % i):
        graph_outputs.restart_timer()
        tokens, lengths = dataset.get_tokens(indices)
        feed_dict = dataset.set_placeholders(indices)
        scores, probabilities = sess.run([score_tensors, probability_tensors], feed_dict=feed_dict)
        predictions = graph_outputs.probs_to_preds(probabilities, lengths)
        tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.output_vocabs})
        graph_outputs.cache_predictions(tokens, indices)
        graph_outputs.update_history(scores)
    graph_outputs.print_recent_history()

    with Timer('Dumping predictions'):
      if output_dir is None and output_filename is None:
        graph_outputs.print_current_predictions()
      else:
        input_dir, input_filename = os.path.split(input_filename)
        if output_dir is None:
          output_dir = os.path.join(self.save_dir, 'parsed', input_dir)
        elif output_filename is None:
          output_filename = input_filename
        
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, output_filename)
        with codecs.open(output_filename, 'w', encoding='utf-8') as f:
          graph_outputs.dump_current_predictions(f)
    if print_time:
      #print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - graph_outputs.time))
      print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - start_time))
    return

  #=============================================================
  def parse_file_ensemble(self, dataset, graph_outputs, sess, saver, output_dir=None, output_filename=None, print_time=True):
    """"""

    probability_tensors = graph_outputs.probabilities
    input_filename = dataset.conllu_files[0]
    graph_outputs.restart_timer()
    collects = []
    for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
      with Timer('Parsing batch %d' % i):
        tokens, lengths = dataset.get_tokens(indices)
        feed_dict = dataset.set_placeholders(indices)
        probabilities = sess.run(probability_tensors, feed_dict=feed_dict)
        collect = {'indices':indices, 'tokens':tokens, 'lengths':lengths, 'probs':probabilities}
        collects.append(collect)

    for n, save_dir in enumerate(self.other_save_dirs):
      print ("### Loading model {} for predicting ###".format(n+1))
      saver.restore(sess, tf.train.latest_checkpoint(save_dir))
      for i, collect in enumerate(collects):
        with Timer('Parsing batch %d' % i):
          feed_dict = dataset.set_placeholders(collect['indices'])
          probabilities = sess.run(probability_tensors, feed_dict=feed_dict)
          for field in probabilities:
            collect['probs'][field] += probabilities[field]

    for i, collect in enumerate(collects):
      for field in collect['probs']:
        collect['probs'][field] /= len(self.other_save_dirs)+1
      with Timer('Merging batch %d' % i):
        predictions = graph_outputs.probs_to_preds(collect['probs'], collect['lengths'])
        collect['tokens'].update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.output_vocabs})
        graph_outputs.cache_predictions(collect['tokens'], collect['indices'])


    with Timer('Dumping predictions'):
      if output_dir is None and output_filename is None:
        graph_outputs.print_current_predictions()
      else:
        input_dir, input_filename = os.path.split(input_filename)
        if output_dir is None:
          output_dir = os.path.join(self.save_dir, 'parsed', input_dir)
        elif output_filename is None:
          output_filename = input_filename
        
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, output_filename)
        with codecs.open(output_filename, 'w', encoding='utf-8') as f:
          graph_outputs.dump_current_predictions(f)
    if print_time:
      print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - graph_outputs.time))
    return

  #=============================================================
  def parse_files(self, dataset, graph_outputs, sess, output_dir=None, print_time=True):
    """"""

    probability_tensors = graph_outputs.probabilities
    graph_outputs.restart_timer()
    for input_filename in dataset.conllu_files:
      for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
        with Timer('batch {}'.format(i)):
          tokens, lengths = dataset.get_tokens(indices)
          feed_dict = dataset.set_placeholders(indices)
          probabilities = sess.run(probability_tensors, feed_dict=feed_dict)
          predictions = graph_outputs.probs_to_preds(probabilities, lengths)
          tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.output_vocabs})
          graph_outputs.cache_predictions(tokens, indices)

      input_dir, input_filename = os.path.split(input_filename)
      if output_dir is None:
        file_output_dir = os.path.join(self.save_dir, 'parsed', input_dir)
      else:
        file_output_dir = output_dir
      if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)
      output_filename = os.path.join(file_output_dir, input_filename)
      with codecs.open(output_filename, 'w', encoding='utf-8') as f:
        graph_outputs.dump_current_predictions(f)
      
      # Load the next conllu file
      dataset.load_next()
    
    if print_time:
      n_files = len(dataset.conllu_files)
      print('\033[92mParsing {} file{} took {:0.1f} seconds\033[0m'.format(n_files, 's' if n_files > 1 else '', time.time() - graph_outputs.time))
    return

  #=============================================================
  def get_input_tensor(self, outputs, reuse=True):
    """"""

    output_keep_prob = 1. if reuse else self.output_keep_prob
    for output in outputs:
      pass # we just need to grab one
    layer = output['recur_layer']
    with tf.variable_scope(self.classname):
      layer = classifiers.hiddens(layer, self.output_size,
                                  hidden_func=self.output_func,
                                  hidden_keep_prob=output_keep_prob,
                                  reuse=reuse)
    return [layer]

  #=============================================================
  @property
  def train_conllus(self):
    return self._config.getfiles(self, 'train_conllus')
  @property
  def aux_conllus(self):
    return self._config.getfiles(self, 'aux_conllus')
  @property
  def cuda_visible_devices(self):
    return os.getenv('CUDA_VISIBLE_DEVICES')
  @property
  def save_dir(self):
    return self._config.getstr(self, 'save_dir')
  @property
  def other_save_dirs(self):
    return self._config.getlist(self, 'other_save_dirs')
  @property
  def vocabs(self):
    return self._vocabs
  @property
  def id_vocab(self):
    return self._id_vocab
  @property
  def input_vocabs(self):
    return self._input_vocabs
  @property
  def throughput_vocabs(self):
    return self._throughput_vocabs
  @property
  def output_vocabs(self):
    return self._output_vocabs
  @property
  def input_networks(self):
    return self._input_networks
  @property
  def input_network_classes(self):
    return self._config.getlist(self, 'input_network_classes')
  @property
  def input_vocab_classes(self):
    return self._config.getlist(self, 'input_vocab_classes')
  @property
  def output_vocab_classes(self):
    return self._config.getlist(self, 'output_vocab_classes')
  @property
  def throughput_vocab_classes(self):
    return self._config.getlist(self, 'throughput_vocab_classes')
  @property
  def l2_reg(self):
    return self._config.getfloat(self, 'l2_reg')
  @property
  def input_size(self):
    return self._config.getint(self, 'input_size')
  @property
  def recur_size(self):
    return self._config.getint(self, 'recur_size')
  @property
  def n_layers(self):
    return self._config.getint(self, 'n_layers')
  @property
  def first_layer_conv_width(self):
    return self._config.getint(self, 'first_layer_conv_width')
  @property
  def conv_width(self):
    return self._config.getint(self, 'conv_width')
  @property
  def input_keep_prob(self):
    return self._config.getfloat(self, 'input_keep_prob')
  @property
  def conv_keep_prob(self):
    return self._config.getfloat(self, 'conv_keep_prob')
  @property
  def recur_keep_prob(self):
    return self._config.getfloat(self, 'recur_keep_prob')
  @property
  def recur_include_prob(self):
    return self._config.getfloat(self, 'recur_include_prob')
  @property
  def bidirectional(self):
    return self._config.getboolean(self, 'bidirectional')
  @property
  def input_func(self):
    input_func = self._config.getstr(self, 'input_func')
    if hasattr(nonlin, input_func):
      return getattr(nonlin, input_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, input_func))
  @property
  def hidden_func(self):
    hidden_func = self._config.getstr(self, 'hidden_func')
    if hasattr(nonlin, hidden_func):
      return getattr(nonlin, hidden_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
  @property
  def recur_func(self):
    recur_func = self._config.getstr(self, 'recur_func')
    if hasattr(nonlin, recur_func):
      return getattr(nonlin, recur_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, recur_func))
  @property
  def highway_func(self):
    highway_func = self._config.getstr(self, 'highway_func')
    if hasattr(nonlin, highway_func):
      return getattr(nonlin, highway_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, highway_func))
  @property
  def recur_cell(self):
    recur_cell = self._config.getstr(self, 'recur_cell')
    if hasattr(recurrent, recur_cell):
      return getattr(recurrent, recur_cell)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(recurrent.__name__, recur_cell))
  @property
  def cifg(self):
    return self._config.getboolean(self, 'cifg')
  @property
  def bilin(self):
    return self._config.getboolean(self, 'bilin')
  @property
  def switch_optimizers(self):
    return self._config.getboolean(self, 'switch_optimizers')
  @property
  def highway(self):
    return self._config.getboolean(self, 'highway')
  @property
  def print_every(self):
    return self._config.getint(self, 'print_every')
  @property
  def max_steps(self):
    return self._config.getint(self, 'max_steps')
  @property
  def max_steps_without_improvement(self):
    return self._config.getint(self, 'max_steps_without_improvement')
  @property
  def n_passes(self):
    return self._config.getint(self, 'n_passes')
  @property
  def parse_devset(self):
    return self._config.getboolean(self, 'parse_devset')
  @property
  def save_model_after_improvement(self):
    return self._config.getboolean(self, 'save_model_after_improvement')
  @property
  def save_model_after_training(self):
    return self._config.getboolean(self, 'save_model_after_training')
  @property
  def classname(self):
    return self.__class__.__name__
  @property
  def share_layer(self):
    return self._config.getboolean(self, 'share_layer')
  @property
  def cpu_num(self):
    return self._config.getint(self, 'cpu_num')
  @property
  def intra_threads(self):
    return self._config.getint(self, 'intra_threads')
  @property
  def inter_threads(self):
    return self._config.getint(self, 'inter_threads')
  @property
  def mix_rates(self):
    return self._config.getlist(self, 'mix_rates')
  @property
  def mix_rate_upbounds(self):
    return self._mix_rate_upbounds
  @property
  def aux_label(self):
    return self._config.getboolean(self, 'aux_label')
