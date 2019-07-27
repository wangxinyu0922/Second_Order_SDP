from __future__ import absolute_import

from parser.structs.vocabs.index_vocabs import IDIndexVocab, DepheadIndexVocab, SemheadGraphIndexVocab
from parser.structs.vocabs.token_vocabs import FormTokenVocab, LemmaTokenVocab, UPOSTokenVocab, XPOSTokenVocab, DeprelTokenVocab, SemrelGraphTokenVocab, SemrelGraphLabelVocab
from parser.structs.vocabs.feature_vocabs import LemmaFeatureVocab, XPOSFeatureVocab, UFeatsFeatureVocab
from parser.structs.vocabs.subtoken_vocabs import FormSubtokenVocab, LemmaSubtokenVocab, UPOSSubtokenVocab, XPOSSubtokenVocab, DeprelSubtokenVocab
from parser.structs.vocabs.pretrained_vocabs import FormPretrainedVocab, LemmaPretrainedVocab, UPOSPretrainedVocab, XPOSPretrainedVocab, DeprelPretrainedVocab
from parser.structs.vocabs.multivocabs import FormMultivocab, LemmaMultivocab, UPOSMultivocab, XPOSMultivocab, XPOSMultivocab, DeprelMultivocab
#our vocab
from parser.structs.vocabs.second_order_vocab import SecondOrderGraphIndexVocab
from parser.structs.vocabs.second_order_token_vocab import SecondOrderGraphTokenVocab
from parser.structs.vocabs.second_order_LBP_vocab import SecondOrderGraphLBPVocab
