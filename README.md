# bert-parser
BERT-based fine-tuning parser

## Package structure
If you want to modify the source code for this directory, here's how it's structured:
* `main.py` The file that runs everything.
* `config/` The config files. Includes config files to replicate the hyperparameters of previous systems.
* `parser/` Where pretty much everything you care about is.
    * `config.py` Extends `SafeConfigParser`. Stores configuration settings.
    * `base_network.py` Contains architecture-neutral code for training networks.
    * `*_network.py` The tensorflow architecture for different kinds of networks.
    * `graph_outputs.py` An incomprehensible mish-mash of stuff. Designed to help printing performance/storing predictions/tracking history without making `base_network.py` or `conllu_dataset.py` too incomprehensible, but could probably be migrated to those two. Or at least cleaned up. I'm sorry.
    * `structs/` Contains data structures that store information.
        * `conllu_dataset.py` Used to manage a single dataset (e.g. trainset, devset, testset) or group of files.
        * `vocabs/` Stores the different kind of vocabularies.
            * `conllu_vocabs.py` Used for subclassing different vocabs so they know which column of the conllu file to read from
            * `base_vocabs.py` Methods and functions relevant to all vocabularies.
            * `index_vocabs.py` Used to store simple indices, such as the index of the token in the sentence of the index of the token's head. (Includes semantic dependency stuff)
            * `token_vocabs.py` Stores discrete, string-valued tokens (words, tags, dependency relations). (Includes semantic dependency stuff)
            * `subtoken_vocabs.py` Stores sequences of discrete subtokens (where the character-level things happen).
            * `pretrained_vocabs.py` For managing word2vec or glove embeddings.
            * `multivocabs.py` Used for managing networks that use more than one kind of word representation (token, subtoken, pretrained)
            * `feature_vocabs.py` For Universal Features and some types of composite POS tags.
        * `buckets/` Code for grouping sequences by length.
            * `dict_multibucket.py` For each vocab, contains a two-column array representing each sentence in the file (in order), where the first column points to a bucket and the second points to an index in the bucket
            * `dict_bucket.py` A bucket that contains all sequences of a certain length range for a given vocab.
            * `list_multibucket.py/list_bucket.py` The bucketing system for character vocabs. Same basic idea but not keyed by vocabulary.
    * `neural/` Where I dump tensorflow code.
        * `nn.py` Tensorflow is great but sometimes it's dumb and I need to fix it.
        * `nonlin.py` All the nonlinearities I might want to try stored in the same file.
        * `embeddings.py` Home-brewn TF functions for embeddings.
        * `recurrent.py` Tensorflow's built-in RNN classes aren't flexible enough for me so I wrote my own.
        * `classifiers.py` TF functions for doing linear/bilinear/attentive transformations.
        * `optimizers.py/` My own handcrafted optimizers. Builds in gradient clipping and doesn't update weights with `nan` gradients. 
            * `optimizer.py` The basic stuff. I'll probably rename it to `base_optimizer.py` for consistency.
            * `adam_optimizer.py` Implements Adam. Renames beta1 and beta2 to mu and nu, and adds in a gamma hp to allow for Nesterov momentum (which probably doesn't actually do anything)
            * `amsgrad_optimizer.py` Implements the AMSGrad fix for Adam. Unlike Adam, AMSGrad struggles when the network is poorly initialized--so if you want to use this in the code you have to set `switch_optimizers` to `True`. This will kick off training with Adam and then switch to AMSGrad after the network has gotten to a point where the gradients aren't so large they break it.
* `scripts/` Where I dump random scripts I write and also Chu-Liu/Edmonds.
* `debug/` Contains a simple little debugging tool that computes how much time and memory a block of code took.
* `hpo/` Some hyperparameter tuning algorithms.
