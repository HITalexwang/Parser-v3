# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
from parser_model.neural import nn, classifiers, nonlin

class EasyFirstTransformerConfig(object):
  """Configuration for `GraphTransformer`."""

  def __init__(self,
               hidden_size=512,
               num_hidden_layers=6,
               num_attention_heads=6,
               intermediate_size=1024,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               acc_mask_dropout_prob=0.1,
               max_position_embeddings=512,
               initializer_range=0.02,
               supervision='graph-bi',
               smoothing_rate=0,
               rm_prev_tp=False,
               num_sup_heads=1,
               n_top_heads=4,
               use_biaffine=True,
               arc_hidden_size=512,
               arc_hidden_add_linear=True,
               arc_hidden_keep_prob=0.67,
               rel_hidden_size=512,
               rel_hidden_add_linear=True,
               rel_hidden_keep_prob=0.67,
               sample_policy='random',
               share_attention_params=True,
               maskout_fully_generated_sents=False,
               use_prob_for_sup=False):
    """Constructs GraphTransformerConfig.

    Args:
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      supervision: The supervision type. (i.e. `direct` or `mask` or `none`)
      smoothing_rate: smoothing rate for the sigmoid cross-entropy
      rm_prev_tp: remove previous true positive predictions from the gold graph ?
      num_sup_heads: number of supervised attention heads
      n_top_heads: number of chosen heads for each layer
    """
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.acc_mask_dropout_prob = acc_mask_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.supervision = supervision
    self.smoothing_rate = smoothing_rate
    self.rm_prev_tp = rm_prev_tp

    self.num_sup_heads = num_sup_heads
    self.n_top_heads = n_top_heads
    self.use_biaffine = use_biaffine
    self.arc_hidden_size = arc_hidden_size
    self.rel_hidden_size = rel_hidden_size
    self.arc_hidden_add_linear = arc_hidden_add_linear
    self.arc_hidden_keep_prob = arc_hidden_keep_prob
    self.sample_policy = sample_policy
    self.share_attention_params = share_attention_params
    self.maskout_fully_generated_sents = maskout_fully_generated_sents
    self.use_prob_for_sup = use_prob_for_sup

    print ("supervision type: {}\nsample policy: {}\nshare attention parameters: {}".format(
            supervision, sample_policy, share_attention_params))
    print ("mask out fully generated sentences:{}\nuse probability for supervision matrix while predicting:{}".format(
            maskout_fully_generated_sents, use_prob_for_sup))

    assert supervision in ['easy-first', 'mask-bi', 'mask-uni', 'none', 'graph-bi', 'graph-uni']

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `GraphTransformerConfig` from a Python dictionary of parameters."""
    config = GraphTransformerConfig()
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `GraphTransformerConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class EasyFirstTransformer(object):
  """

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.GraphTransformerConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.GraphTransformer(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_tensor,
               input_mask=None,
               unlabeled_targets=None,
               null_mask=None,
               scope=None):
    """Constructor for GraphTransformer.

    Args:
      config: `GraphTransformerConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_tensors: float32 Tensor of shape [batch_size, seq_length, embedding_size].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    config.is_training = is_training
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
      config.acc_mask_dropout_prob = 0.0
      config.arc_hidden_keep_prob = 1.0

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
    if unlabeled_targets is not None:
      if config.supervision.startswith('easy-first'):
        assert isinstance(unlabeled_targets, tf.Tensor)
      elif len(unlabeled_targets) != config.num_hidden_layers:
        raise ValueError(
          "The number of accessible matrices does not match the number of attention layers.")
    elif config.supervision != 'none':
      raise ValueError(
          "Input accessible matrix to the transformer is None, while the supervision is not none.")

    with tf.variable_scope(scope, default_name="transformer"):
      with tf.variable_scope("embeddings"):
        # project input tensor to hidden_size
        input_tensor = tf.layers.dense(
          input_tensor,
          config.hidden_size,
          kernel_initializer=create_initializer(config.initializer_range))

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        # shape = [batch_size, seq_length, embedding_size]
        self.embedding_output = embedding_postprocessor(
            input_tensor=input_tensor,
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        if len(get_shape_list(input_mask)) == 3:
          attention_mask = input_mask
        else:
          attention_mask = create_attention_mask_from_input_mask(
              input_tensor, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers, self.outputs = easy_first_transformer_model(
            config=config,
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            null_mask=null_mask,
            unlabeled_targets=unlabeled_targets,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            acc_mask_dropout_prob=config.acc_mask_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            supervision=config.supervision)

      self.sequence_output = self.all_encoder_layers[-1]

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_outputs(self, do_eval_by_layer=False):
    return self.compute_outputs(self.outputs,do_eval_by_layer=do_eval_by_layer)

  def compute_outputs(self, outputs, do_eval_by_layer=False):

    # n_layers x [batch_size, seq_len, seq_len]
    predictions = outputs['predictions']
    # [batch_size, seq_len, seq_len]
    predictions = nn.greater(tf.add_n(predictions),0)
    outputs['preds_by_layer'] = outputs['predictions']
    outputs['predictions'] = predictions
    
    targets = outputs['unlabeled_targets']
    n_targets = tf.reduce_sum(targets)
    if do_eval_by_layer:
      outputs['n_acc_true_positives'] = []
      outputs['n_acc_false_positives'] = []
      outputs['n_acc_false_negatives'] = []
      #outputs['acc_loss_by_layer'] = []
      # [B, F, T] * [B, F, T] -> [B, F, T]
      for n_layer, preds in enumerate(outputs['preds_by_layer']):
        true_positives = preds * targets
        # [B, F, T] -> ()
        n_predictions = tf.reduce_sum(preds)
        
        n_true_positives = tf.reduce_sum(true_positives)
        n_false_positives = n_predictions - n_true_positives
        n_false_negatives = n_targets - n_true_positives
        outputs['n_acc_true_positives'].append(n_true_positives)
        outputs['n_acc_false_positives'].append(n_false_positives)
        outputs['n_acc_false_negatives'].append(n_false_negatives)
        #outputs['acc_loss_by_layer'].append(outputs['acc_loss'][n_layer])

    true_positives = predictions * targets
    # [B, F, T] -> ()
    n_predictions = tf.reduce_sum(predictions)
    #n_targets = tf.reduce_sum(targets)

    n_true_positives = tf.reduce_sum(true_positives)
    n_false_positives = n_predictions - n_true_positives
    n_false_negatives = n_targets - n_true_positives

    # (n x m x m) -> (n)
    n_targets_per_sequence = tf.reduce_sum(targets, axis=[1,2])
    n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
    # (n) x 2 -> ()
    n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

    outputs['n_unlabeled_true_positives'] = n_true_positives
    outputs['n_unlabeled_false_positives'] = n_false_positives
    outputs['n_unlabeled_false_negatives'] = n_false_negatives
    outputs['true_positives'] = true_positives
    outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
    return outputs
    

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  #def get_embedding_table(self):
  #  return self.embedding_table


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  if use_one_hot_embeddings:
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def attention_layer(from_tensor,
                    to_tensor,
                    config,
                    attention_mask=None,
                    null_mask=None,
                    remained_unlabeled_targets=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    acc_mask_dropout_prob=0.1,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    supervision='easy-first'):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    null_mask: int32 Tensor of shape [batch_size, from_seq_length].
      The position of the NULL token is 1, all others are 0.
    remained_unlabeled_targets: Tensor of shape [batch_size, seq_len, seq_len]. 
      Supervision for the mask of the attention probs.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
    supervision: supervision type (i.e. `direct` or `mask`)
    reuse_weight: whether to reuse attention weights across layers
    num_sup_heads: number of supervised attention heads
    n_top_heads: number of selected head at this layer
    sup_head_size: size of supervised attention head

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # Standard Attention
  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask_ = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask_, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  outputs = {}

  with tf.variable_scope("graph_attention"):
    # losses: S * []
    # predictions: S * [B, F]
    # graph_context_layer: [B, F, n*h]
    losses, predictions, probabilities, graph_context_layer, used_heads, allowed_heads = easy_first_one_step(
                                          config,
                                          remained_unlabeled_targets, from_tensor_2d,
                                          to_tensor_2d, attention_mask, null_mask, batch_size,
                                          from_seq_length, to_seq_length, 
                                          policy=config.sample_policy,
                                          use_biaffine=config.use_biaffine,
                                          arc_hidden_size = config.arc_hidden_size, rel_hidden_size = 512,
                                          num_sup_heads=config.num_sup_heads, 
                                          n_top_heads=config.n_top_heads,
                                          smoothing_rate=config.smoothing_rate,
                                          initializer_range=initializer_range,
                                          add_linear=config.arc_hidden_add_linear,
                                          arc_hidden_keep_prob=config.arc_hidden_keep_prob,
                                          do_return_2d_tensor=do_return_2d_tensor)
    outputs['acc_loss'] = tf.add_n(losses)
    outputs['predictions'] = predictions
    outputs['probabilities'] = probabilities
    outputs['used_heads'] = used_heads
    outputs['allowed_heads'] = allowed_heads

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
    context_layer = tf.concat([context_layer, graph_context_layer], axis=-1)
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])
    context_layer = tf.concat([context_layer, graph_context_layer], axis=-1)

  return context_layer, outputs

def easy_first_one_step(config, remained_unlabeled_targets, from_tensor_2d, to_tensor_2d,
                      attention_mask, null_mask, batch_size, from_seq_length, 
                      to_seq_length, use_biaffine=False, 
                      arc_hidden_size=512, rel_hidden_size=512,
                      num_sup_heads=1, n_top_heads=4, policy='random',
                      smoothing_rate=0.1, initializer_range=0.02,
                      add_linear=True, arc_hidden_keep_prob=0.67,
                      do_return_2d_tensor=True):
  """
  remained_unlabeled_targets: [B, F, T], the arcs that have not been generated yet
  attention_scores: [B, N, F, T], N attention heads
  null_mask: [B, F], the null token is 1, all others are 0
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   n = `num_sup_heads`
  #   h = `arc_hidden_size`

  # `query_layer` = [B*F, n*h], works as representation of dependent token
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_sup_heads * arc_hidden_size,
      name="graph_query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, n*h], works as representation of parent token
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_sup_heads * arc_hidden_size,
      name="graph_key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, n*h]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_sup_heads * arc_hidden_size,
      name="graph_value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, n, F, h]
  query_layer = transpose_for_scores(query_layer, batch_size, num_sup_heads,
                                    from_seq_length, arc_hidden_size)

  # `key_layer` = [B, n, T, h]
  key_layer = transpose_for_scores(key_layer, batch_size, num_sup_heads,
                                   to_seq_length, arc_hidden_size)

  if use_biaffine:
    #query_layer = query_layer[:,0,:,:]
    #key_layer = key_layer[:,0,:,:]
    with tf.variable_scope("Unlabeled"):
      # `attention_scores` = [B, n, F, T]
      attention_scores, _ = classifiers.bilinear_attention(
                              query_layer, key_layer,
                              hidden_keep_prob=arc_hidden_keep_prob,
                              add_linear=add_linear)
    #attention_scores = tf.expand_dims(attention_scores, 1)
  else:
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, n, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                  1.0 / math.sqrt(float(arc_hidden_size)))

  use_null_mask = True
  # [B, F, T] -> [B, F], use the second column, 
  # since the first column is all 0 padding for null token
  if use_null_mask:
    sliced_attention_mask = attention_mask[:,:,1] + null_mask
  else:
    sliced_attention_mask = attention_mask[:,:,1]

  batch_size, _, from_seq_length, to_seq_length = get_shape_list(attention_scores)
  # [B, 1, F] + [B, F, T] -> [B, F, T] , allowed heads including NULL
  #if use_null_mask:
  #  allowed_heads = tf.expand_dims(null_mask, axis=1) + remained_unlabeled_targets
  #else:
  allowed_heads = remained_unlabeled_targets
  losses = []
  predictions = []
  supervised_probs = []
  probabilities = []
  used_heads = []
  for i in range(num_sup_heads):
    # [B, F, T]
    mask_with_null = attention_mask + tf.expand_dims(null_mask, axis=1)
    supervised_logits = attention_scores[:,i,:,:]
    #probability = tf.nn.softmax(supervised_logits) * tf.to_float(mask_with_null)
    if policy == 'top_k':
      # [B, F, T], 1 for selected head, 0 otherwise
      selected_gold_heads_3D = top_k_heads(supervised_logits, allowed_heads, null_mask, n=n_top_heads)
      probability = tf.nn.sigmoid(supervised_logits) * tf.to_float(mask_with_null)
    elif policy == 'random':
      selected_gold_heads = random_sample(allowed_heads, null_mask, from_seq_length)
      probability = tf.nn.softmax(supervised_logits) * tf.to_float(mask_with_null)
    elif policy == 'confidence':
      selected_gold_heads = confidence_sample(supervised_logits, allowed_heads, null_mask, from_seq_length)
      probability = tf.nn.softmax(supervised_logits) * tf.to_float(mask_with_null)
    probabilities.append(probability)
    if config.maskout_fully_generated_sents:
      # [B, 1], the number of remained arcs of each sentence
      remained_arcs_cnt = tf.reduce_sum(tf.reduce_sum(remained_unlabeled_targets, axis=-1), axis=-1, keep_dims=True)
      # [B, F]
      zero_mask = tf.zeros_like(sliced_attention_mask)
      # [B, F], tile it to 2D
      remained_arcs_cnt_ = remained_arcs_cnt + zero_mask
      # [B, F]
      new_attention_mask = tf.where(tf.equal(remained_arcs_cnt_,0), zero_mask, sliced_attention_mask)
    else:
      new_attention_mask = sliced_attention_mask
    
    if policy == 'top_k':
      add_null_for_words_with_no_head = True
      # [B, F, T]
      zeros = tf.zeros_like(selected_gold_heads_3D)
      # [B, F, T]
      null_mask_3D = tf.expand_dims(null_mask, axis=1) + zeros
      if add_null_for_words_with_no_head:
        # [B, F, 1], the number of selected heads for each word
        selected_heads_cnt = tf.reduce_sum(selected_gold_heads_3D, axis=-1, keep_dims=True)
        # [B, F, T]
        selected_heads_cnt_3D = selected_heads_cnt + zeros
        # [B, F, T]
        selected_gold_heads_3D_ = tf.where(tf.equal(selected_heads_cnt_3D,0), null_mask_3D, selected_gold_heads_3D)
      else:
        selected_gold_heads_3D_ = selected_gold_heads_3D
      used_heads.append(selected_gold_heads_3D_)

      if use_null_mask:
        attention_mask_3D = attention_mask + null_mask_3D
      else:
        attention_mask_3D = attention_mask
      loss = tf.losses.sigmoid_cross_entropy(selected_gold_heads_3D_, supervised_logits,
                                              weights=attention_mask_3D,
                                              label_smoothing=smoothing_rate)
      losses.append(loss)
      # [B, F, T] -> [B, F, T]
      prediction = nn.greater(supervised_logits, 0, dtype=tf.int32) * attention_mask
      predictions.append(prediction)

      #print ('Is training:',config.is_training)
      if config.is_training:
        # [B, F, T]
        smoothed_head_tensor = tf.to_float(selected_gold_heads_3D_) * (1 - smoothing_rate) + 0.5 * smoothing_rate
        supervised_probs.append(smoothed_head_tensor)
      else:
        if config.use_prob_for_sup:
          # probability: [B, F, T]
          supervised_probs.append(probability)
        else:
          smoothed_head_tensor = prediction
          if smoothing_rate > 0:
            smoothed_head_tensor = tf.cast(smoothed_head_tensor, supervised_logits.dtype)
            smoothed_head_tensor = (smoothed_head_tensor * (1 - smoothing_rate) + 0.5 * smoothing_rate)
          supervised_probs.append(smoothed_head_tensor)
    # for random & confidence
    else:
      used_heads.append(selected_gold_heads)
      # [B, F], [B, F, T], [B, F] -> ()
      loss = tf.losses.sparse_softmax_cross_entropy(selected_gold_heads, supervised_logits, 
                                                    weights=new_attention_mask)
      losses.append(loss)
      # [B, F], the real prediction of attention head-i
      prediction = tf.argmax(supervised_logits, axis=-1, output_type=tf.int32)
      predictions.append(prediction)

      #print ('Is training:',config.is_training)
      if config.is_training:
        # [B, F, T], expand the selected heads to 3D
        # arc entry = 1 - smoothing_rate, other entry = smoothing_rate
        one_hot_probs = tf.one_hot(selected_gold_heads, to_seq_length, on_value=1.0-smoothing_rate, 
                                  off_value=0.0+smoothing_rate, axis=-1)
        #eyes = tf.eye(to_seq_length)
        #augmented_probs = one_hot_probs * (1-eyes) + eyes
        supervised_probs.append(one_hot_probs)
      else:
        if config.use_prob_for_sup:
          # probability: [B, F, T]
          supervised_probs.append(probability)
        else:
          one_hot_probs = tf.one_hot(prediction, to_seq_length, on_value=1.0-smoothing_rate, 
                                  off_value=0.0+smoothing_rate, axis=-1)
          supervised_probs.append(one_hot_probs)
      

  if num_sup_heads == 1:
    # [B, F, T] -> [B, 1, F, T]
    stacked_supervised_probs = tf.expand_dims(supervised_probs[0], axis=1)
  else:
    # n x [B, F, T] -> [B, n, F, T]
    stacked_supervised_probs = tf.stack(supervised_probs, axis=1)
  # Normalize the attention scores to probabilities.

  # Apply probs to value matrix
  # `value_layer` = [B, T, n, h]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_sup_heads, arc_hidden_size])

  # `value_layer` = [B, n, T, h]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, n, F, h]
  graph_context_layer = tf.matmul(stacked_supervised_probs, value_layer)

  # `context_layer` = [B, F, n, h]
  graph_context_layer = tf.transpose(graph_context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, n*h]
    graph_context_layer = tf.reshape(
        graph_context_layer,
        [batch_size * from_seq_length, num_sup_heads * arc_hidden_size])
  else:
    # `context_layer` = [B, F, n*h]
    graph_context_layer = tf.reshape(
        graph_context_layer,
        [batch_size, from_seq_length, num_sup_heads * arc_hidden_size])

  if policy == 'top_k':
    # S x [B, F, T] -> [B, F, T]
    predictions_ = nn.greater(tf.add_n(predictions), 0) * attention_mask
    # S x [B, F, T] -> [B, F, T]
    used_heads_ = nn.greater(tf.add_n(used_heads), 0) * attention_mask
  else:
    # S x [B, F] -> [B, F, T]
    predictions_ = gather_subgraphs(predictions, attention_mask)
    # S x [B, F] -> [B, F, T]
    used_heads_ = gather_subgraphs(used_heads, attention_mask)

  return losses, predictions_, probabilities, graph_context_layer, used_heads_, allowed_heads

def gather_subgraphs(head_index_list, attention_mask):
  """
  Input:
    head_index_list: n_sup_heads x [B, F]
  Return:
    gathered_graph: [B, F, T], where 1 indicate arc, 0 indicates not arc
  """
  batch_size, seq_len = get_shape_list(head_index_list[0])
  matrices = []
  for i in range(len(head_index_list)):
    # [B, F, T], expand the selected heads to 3D
    one_hot_matrix = tf.one_hot(head_index_list[i], seq_len, on_value=1, off_value=0, axis=-1)
    matrices.append(one_hot_matrix)
  # [B, F, T]
  gathered_graph = nn.greater(tf.add_n(matrices), 0) * attention_mask
  return gathered_graph

def confidence_sample(supervised_logits, allowed_heads, null_mask, seq_len):
  """
  Ramdonly sample one head for each word from the allowed_heads
  Input:
    supervised_logits: [B, F, T], logits (Q*K) for the current attention head
    allowed_heads: [B, F, T], entry is 1 if the head is allowed, elseweise 0
    null_mask: [B, F], entry is 1 if is the null token, elsewise 0
  Return:
    selected_gold_heads: [B, F], each entry is the index of the chosen word
  """
  disallowed_adder = (1.0 - tf.cast(allowed_heads, tf.float32)) * -10000.0
  # [B, F, T], scores for allowed dep heads (including NULL)
  allowed_scores = supervised_logits + disallowed_adder
  # [B, F], the best allowed dep heads at attention head-i 
  selected_heads_ = tf.argmax(allowed_scores, axis=-1, output_type=tf.int32)

  # [B], the null index of each sentence
  null_indices = tf.argmax(null_mask, axis=-1)
  # [B, F]
  zeros = tf.zeros_like(null_mask, dtype=tf.int32)
  # [B, F], tile null_indices to 2D
  null_index_tensor = tf.cast(tf.expand_dims(null_indices, axis=-1), dtype=tf.int32) + zeros
  # [B, F], number of allowed heads for each word
  allowed_head_cnt = tf.reduce_sum(allowed_heads, axis=-1)
  # [B, F], if there is no head allowed, select the null token
  #print (allowed_head_cnt)
  selected_gold_heads = tf.where(tf.equal(allowed_head_cnt,0), null_index_tensor, selected_heads_)

  #debug_tensor = tf.concat([allowed_head_cnt, null_index_tensor,
  # selected_heads_, selected_gold_heads], axis=-1)

  return selected_gold_heads

def random_sample(allowed_heads, null_mask, seq_len):
  """
  Ramdonly sample one head for each word from the allowed_heads
  Input:
    allowed_heads: [B, F, T], entry is 1 if the head is allowed, elseweise 0
    null_mask: [B, F], entry is 1 if is the null token, elsewise 0
  Return:
    selected_gold_heads: [B, F], each entry is the index of the chosen word
  """
  # [B], the null index of each sentence
  null_indices = tf.argmax(null_mask, axis=-1)
  # [B, F]
  zeros = tf.zeros_like(null_mask, dtype=tf.int32)
  # [B, F], tile null_indices to 2D
  null_index_tensor = tf.cast(tf.expand_dims(null_indices, axis=-1), dtype=tf.int32) + zeros
  # [B, F], number of allowed heads for each word
  allowed_head_cnt = tf.reduce_sum(allowed_heads, axis=-1)
  # [B, F, T], uniform mask in [0, 1)
  random_mask = tf.random_uniform(shape=tf.shape(allowed_heads))
  # [B, F], the randomly selected head index
  selected_heads_ = tf.cast(tf.argmax(random_mask * tf.to_float(allowed_heads), axis=-1), dtype=tf.int32)
  # [B, F], if there is no head allowed, select the null token
  #print (allowed_head_cnt)
  selected_gold_heads = tf.where(tf.equal(allowed_head_cnt,0), null_index_tensor, selected_heads_)

  #debug_tensor = tf.concat([allowed_head_cnt, null_index_tensor,
  # selected_heads_, selected_gold_heads], axis=-1)

  return selected_gold_heads

def top_k_heads(supervised_logits, allowed_heads, null_mask, n=4):
  """
  Input:
    scores: [B, F, T], the score matrix
    k: the number of top heads to return
    null_mask: [B, F], entry is 1 if is the null token, elsewise 0
  Return:
    #selected_gold_heads: [B, F], each entry is the index of the chosen word
    selected_gold_heads_3D: [B, F, T], 1 represent the chosen head, 0 otherwise
  """
  disallowed_adder = (1.0 - tf.cast(allowed_heads, tf.float32)) * -10000.0
  # [B, F, T], scores for allowed dep heads (including NULL)
  scores = supervised_logits + disallowed_adder

  batch_size, from_seq_len, to_seq_len = get_shape_list(scores)
  # [B, F, T] -> [B, F*T] (reshape) -> [B, k]
  _, top_indices = tf.nn.top_k(tf.reshape(scores, (batch_size,-1)), n)
  # [B, k, 2], the cordinates of top k arcs
  top_indices = tf.stack(((top_indices // to_seq_len), (top_indices % to_seq_len)), -1)
  # [B, k, 2] -> [B, T(=F), T], 1 represent the chosen head, 0 otherwise
  selected_gold_heads_3D = indices_to_tensor(top_indices, batch_size, n, to_seq_len, value=1)

  zeros = tf.zeros_like(selected_gold_heads_3D)
  # [B, F, T]
  null_mask_3D = tf.expand_dims(null_mask, axis=1) + zeros

  # [B, F, 1], the number of selected heads for each word
  allowed_head_cnt = tf.reduce_sum(allowed_heads, axis=-1, keep_dims=True)
  # [B, F, T]
  allowed_head_cnt_3D = allowed_head_cnt + zeros
  # [B, F, T]
  selected_gold_heads_3D = tf.where(tf.equal(allowed_head_cnt_3D,0), null_mask_3D, selected_gold_heads_3D)

  return selected_gold_heads_3D

def indices_to_tensor(indices, batch_size, k, to_seq_len, value=1):
  """
  Input:
    indices: [B, k, 2], the cordinates of top k arcs
  Return:
    tensor: []
  """
  # add batch id to indices
  # [B, 1, 1]
  batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
  # [B, k, 1]
  batch_index = tf.tile(batch_idx, [1, k, 1])
  # [B, k, 1], [B, k, 2] -> [B, k, 3]
  batched_indices = tf.concat([batch_index, indices], axis=-1)
  # [B*k, 3]
  reshaped_indices = tf.reshape(batched_indices, [-1, 3])

  # [B*k]
  values = tf.ones(shape=[batch_size*k], dtype=tf.int32) * value
  shape = [batch_size, to_seq_len, to_seq_len]
  # [B, T, T]
  tensor = tf.scatter_nd(reshaped_indices, values, shape)
  return tensor


def easy_first_transformer_model(input_tensor,
                      config,
                      attention_mask=None,
                      null_mask=None,
                      unlabeled_targets=None,
                      hidden_size=512,
                      num_hidden_layers=6,
                      num_attention_heads=6,
                      intermediate_size=1024,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      acc_mask_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      supervision='easy-first'):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  outputs = {'acc_loss':[], 'probabilities':[], 'predictions':[], 
            'unlabeled_targets':unlabeled_targets, 'used_heads':[],
            'allowed_heads':[]}


  # here the unlabeled_targets is the graph adjacency matrix
  remained_unlabeled_targets = unlabeled_targets

  for layer_idx in range(num_hidden_layers):
    #print ('layer_id:{}'.format(layer_idx))
    if config.share_attention_params:
      scope = "block"
    else:
      scope = "layer_%d" % layer_idx
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    #with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head, outputs_ = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              config=config,
              attention_mask=attention_mask,
              null_mask=null_mask,
              remained_unlabeled_targets=remained_unlabeled_targets,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              acc_mask_dropout_prob=acc_mask_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,
              supervision=supervision)
          attention_heads.append(attention_head)
          for field in outputs_:
            outputs[field].append(outputs_[field])
          # remove the used heads from supervision
          remained_unlabeled_targets = remained_unlabeled_targets - outputs_['used_heads']
        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs, outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output, outputs


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
