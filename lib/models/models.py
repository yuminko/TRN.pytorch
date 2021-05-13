from .generalized_trn import GeneralizedTRN
from .lstm import baselineLSTM
from .lstm_second import SecondLSTM

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM' : baselineLSTM,
    'Second' : SecondLSTM
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    print('Models:  ', args.model)
    return meta_arch(args)
