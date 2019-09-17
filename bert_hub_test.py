import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

import tensorflow as tf
import tensorflow_hub as hub

# This is a path to an uncased (all lowercase) version of BERT
# BERT_MODEL_HUB = "/Users/longxud/Documents/Code/bert/hub_bert_cased_L-12_H-768_A-12"
BERT_MODEL_HUB = "bert/hub_bert_cased_L-12_H-768_A-12"
# BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""

    bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                              tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


tokenizer = create_tokenizer_from_hub_module()
t = tokenizer.wordpiece_tokenizer.tokenize("This here's an example of using the BERT tokenizer")
t = ['[CLS]'] + t + ['[SEP]']
print(t)
i = tokenizer.convert_tokens_to_ids(t)
print(i)

bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)
bert_inputs = dict(
    input_ids=[i],
    input_mask=[[0] * len(i)],
    segment_ids=[[0] * len(i)]
)
bert_outputs = bert_module(
    inputs=bert_inputs,
    signature="tokens",
    as_dict=True
)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ret = sess.run(bert_outputs)
print(ret)
print(ret['sequence_output'].shape)
print(len())