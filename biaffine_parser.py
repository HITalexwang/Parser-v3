#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import shutil
import sys
import time
import six
from six.moves import input
import codecs
from argparse import ArgumentParser

from parser_model.config import Config
import parser_model

class Biaffine_Parser(object):
  def __init__(self, save_dir, other_save_dirs=None):
    """
    Input Params
    ----------
    save_dir : ``str``, required.
        Path to the main model file.
    other_save_dirs: ``str``, optional (default = None).
        Path to other model files for ensemble, paths should be split with ``:``
    """
    kwargs = {'DEFAULT':{}}
    config_file = os.path.join(save_dir, 'config.cfg')

    kwargs['DEFAULT']['save_dir'] = save_dir
    kwargs['DEFAULT']['other_save_dirs'] = other_save_dirs

    config = Config(defaults_file='', config_file=config_file, **kwargs)
    network_class = config.get('DEFAULT', 'network_class')
    NetworkClass = getattr(parser, network_class)
    self.network = NetworkClass(input_networks=set(), config=config)
    return

  def sentence2conllu(self, sentence):
    """
    Input Params
    ----------
    sentence : List[tuple(``str``,``str``)], required.
        A list of token-pos pair of a sentence, with each tuple consists of (token, POS).
    
    Output Params
    ----------
    conllu_sent: List[tuple(``str``*10)]
        A list of conllu-format tuple for the sentence, each tuple consists of
        (id, token, token, POS, POS, _ * 5). Some of the _ are blank to be predicted.
    """
    conllu_sent = []
    for id, item in enumerate(sentence):
      conllu_sent.append((str(id+1), item[0], item[0], item[1], item[1], '_', '_', '_', '_', '_'))
    return conllu_sent

  def refine_output(self, sentence):
    """
    Input Params
    ----------
    sentence : List[tuple(``str``,``str``,``str``,``str``)], required.
        A list of a sentence, with each tuple consists of (id, token, POS, predctions).
    
    Output Params
    ----------
    conllu_sent: List[tuple(``str``,``str``,``str``,List[(``int``,``str``)])]
        A list of output tuple for the sentence, each tuple consists of
        (id, token, POS, List[(head, relation)]).
        Note that the List is the predicted heads, which may contain more than one head.
    """
    output = []
    for item in sentence:
      preds = [(int(head), rel) for (head, rel) in [head.split(':') for head in item[3].split('|')]]
      output.append((item[0],item[1],item[2],preds))
    return output

  def parse(self, sentences):
    """
    Input Params
    ----------
    sentences : List[List[tuple(``str``,``str``)]], required.
        A list of input sentences, each list consists of a list of token-pos pair of 
        a sentence, with each tuple consists of (token, POS).
    
    Output Params
    ----------
    predictions_ : List[List[tuple(``str``,``str``,``str``,List[(``int``,``str``)])]]
        A list of output sentences, each list consists of the output for a sentence.
        For a sentence the list consists of tuples for tokens in the sentence.
        each tuple consists of (id, token, POS, List[(head, relation)])
        Note that the List is the predicted heads, which may contain more than one head.
    """
    sentences_ = [self.sentence2conllu(sent) for sent in sentences]
    predictions = self.network.parse_wrapper(sentences_)
    predictions_ = [[(tok[0],tok[1],tok[3],tok[8]) for tok in pred] for pred in predictions]
    predictions_ = [self.refine_output(pred) for pred in predictions_]
    return predictions_

#***************************************************************
if __name__ == '__main__':
  """
  Gold Examples:

  1 早起  早起  AD  AD  _ 2 3_Exp 2:3_Exp _
  2 使 使 VV  VV  _ 0 Root  0:Root  _
  3 人 人 NN  NN  _ 2 3_Datv  2:3_Datv|4:3_Exp  _
  4 健康  健康  VV  VV  _ 2 3_eEfft 2:3_eEfft _
  --------------
  1 村庄  村庄  NN  NN  _ 2 3_Exp 2:3_Exp _
  2 惊醒  惊醒  VV  VV  _ 0 Root  0:Root  _
  3 了 了 AS  AS  _ 2 3_mDepd 2:3_mDepd _
  4 。 。 PU  PU  _ 2 3_mPunc 2:3_mPunc _
  """
  argparser = ArgumentParser('Network')
  argparser.add_argument('save_dir', type=str, help='path to the main model (with the embedding)')
  argparser.add_argument('--other_save_dirs', type=str, help='paths to other models (split with ``:``)')
  args = argparser.parse_args()

  #save_dir = '/mnt/hgfs/share/huawei/sdp-char-v0'
  #other_save_dirs = '/mnt/hgfs/share/huawei/sdp-char-v1'
  example = [[("早起","AD"),("使","VV"),("人","NN"),("健康","VV")],
             [("村庄","NN"),("惊醒","VV"),("了","AS"),("。","PU")]]
  parser = Biaffine_Parser(args.save_dir, other_save_dirs=args.other_save_dirs)
  print (parser.parse(example))
