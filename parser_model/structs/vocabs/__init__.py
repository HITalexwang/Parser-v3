from __future__ import absolute_import

from parser_model.structs.vocabs.index_vocabs import IDIndexVocab, DepheadIndexVocab, SemheadGraphIndexVocab
from parser_model.structs.vocabs.token_vocabs import FormTokenVocab, LemmaTokenVocab, UPOSTokenVocab, XPOSTokenVocab, DeprelTokenVocab, SemrelGraphTokenVocab
from parser_model.structs.vocabs.feature_vocabs import LemmaFeatureVocab, XPOSFeatureVocab, UFeatsFeatureVocab
from parser_model.structs.vocabs.subtoken_vocabs import FormSubtokenVocab, LemmaSubtokenVocab, UPOSSubtokenVocab, XPOSSubtokenVocab, DeprelSubtokenVocab
from parser_model.structs.vocabs.pretrained_vocabs import FormPretrainedVocab, LemmaPretrainedVocab, UPOSPretrainedVocab, XPOSPretrainedVocab, DeprelPretrainedVocab
from parser_model.structs.vocabs.multivocabs import FormMultivocab, LemmaMultivocab, UPOSMultivocab, XPOSMultivocab, XPOSMultivocab, DeprelMultivocab
from parser_model.structs.vocabs.bert_vocabs import FormBERTVocab, LemmaBERTVocab, UPOSBERTVocab, XPOSBERTVocab, DeprelBERTVocab
