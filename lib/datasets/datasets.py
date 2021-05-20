from .hdd_data_layer import TRNHDDDataLayer
from .thumos_data_layer import TRNTHUMOSDataLayer
from .thumos_data_layer import LSTMTHUMOSDataLayer

_DATA_LAYERS = {
    'TRNHDD': TRNHDDDataLayer,
    'TRNTHUMOS': TRNTHUMOSDataLayer,
    'LSTMTHUMOS': LSTMTHUMOSDataLayer,
    'LSTMTVSeries': LSTMTHUMOSDataLayer,
    'SecondTHUMOS': LSTMTHUMOSDataLayer,
    'SecondTVSeries':LSTMTHUMOSDataLayer
}

def build_dataset(args, phase):
    print('DATASET: ', args.dataset)
    data_layer = _DATA_LAYERS[args.model + args.dataset]
    return data_layer(args, phase)
