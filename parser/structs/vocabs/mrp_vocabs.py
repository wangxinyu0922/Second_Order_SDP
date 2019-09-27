from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#***************************************************************
class MrpVocab():
  """"""
  
  _field = None
  _n_splits = None
  # _mrp_idx = None
  
  #=============================================================
  @property
  def n_splits(self):
    return self._n_splits
  @property
  def field(self):
    return self._field
  @property
  def mrp_idx(self):
    return self._mrp_idx
  
#***************************************************************
class NodeIDVocab(MrpVocab):
  _field = 'id'
  _mrp_idx = _field
class LabelVocab(MrpVocab):
  _field = 'label'
  _mrp_idx = _field
class AnchorVocab(MrpVocab):
  _field = 'anchor_word'
  _mrp_idx = _field
class SemrelVocab(MrpVocab):
  _field = 'semrel'
  _mrp_idx = _field
class SemheadVocab(MrpVocab):
  _field = 'semhead'
  _mrp_idx = 'semrel'
class WordVocab(MrpVocab):
  _field = 'correspond_word'
  _mrp_idx = _field
class SrcCopyMapVocab(MrpVocab):
  _field = 'src_copy_map'
  _mrp_idx = _field
class TgtCopyMapVocab(MrpVocab):
  _field = 'tgt_copy_map'
  _mrp_idx = _field
class SrcCopyIndicesVocab(MrpVocab):
  _field = 'src_copy_indices'
  _mrp_idx = _field
class TgtCopyIndicesVocab(MrpVocab):
  _field = 'tgt_copy_indices'
  _mrp_idx = _field