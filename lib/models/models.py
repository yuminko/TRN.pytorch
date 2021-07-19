from .generalized_trn import GeneralizedTRN
from .lstm import baselineLSTM
from .lstm_second import SecondLSTM
from .lstm_delta import DeltaLSTM

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM' : baselineLSTM,
    'Second' : SecondLSTM,
    'Delta' : DeltaLSTM
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    print('Models:  ', args.model)
    return meta_arch(args)
