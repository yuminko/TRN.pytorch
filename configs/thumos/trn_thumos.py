from configs import parse_base_args, build_data_info

__all__ = ['parse_trn_args', 'parse_lstm_args', 'parse_second_args']

def parse_trn_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='/data/yumin/THUMOS', type=str)
    parser.add_argument('--model', default='TRN', type=str)
    parser.add_argument('--inputs', default='multistream', type=str)
    parser.add_argument('--hidden_size', default=4096, type=int)
    parser.add_argument('--camera_feature', default='resnet200-fc', type=str)
    parser.add_argument('--motion_feature', default='bn_inception', type=str)
    parser.add_argument('--enc_steps', default=16, type=int)
    parser.add_argument('--dec_steps', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    return build_data_info(parser.parse_args())

def parse_lstm_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='/data/yumin/THUMOS', type=str)
    parser.add_argument('--model', default='LSTM', type=str)
    parser.add_argument('--inputs', default='multistream', type=str)
    parser.add_argument('--hidden_size', default=4096, type=int)
    parser.add_argument('--camera_feature', default='resnet200-fc', type=str)
    parser.add_argument('--motion_feature', default='bn_inception', type=str)
    parser.add_argument('--enc_steps', default=64, type=int)
    return build_data_info(parser.parse_args())

def parse_second_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='/data/yumin/THUMOS', type=str)
    parser.add_argument('--model', default='Delta', type=str)
    parser.add_argument('--inputs', default='multistream', type=str)
    parser.add_argument('--hidden_size', default=4096, type=int)
    parser.add_argument('--camera_feature', default='resnet200-fc', type=str)
    parser.add_argument('--motion_feature', default='bn_inception', type=str)
    parser.add_argument('--enc_steps', default=64, type=int)
    # parser.add_argument('--step_size',  action='append', help='get list of step size')
    parser.add_argument('--dirichlet', action='store_true')
    parser.add_argument('--method', default=None, type=str, help=['Mean','Sample'])
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--var_method', default=None, type=str, help=['covariance','diagonal'])
    return build_data_info(parser.parse_args())
