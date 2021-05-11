from .generalized_trn import GeneralizedTRN
from .lstm import baselineLSTM
from .model_two import SecondModel

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM' : baselineLSTM,
    'Second': SecondModel
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
