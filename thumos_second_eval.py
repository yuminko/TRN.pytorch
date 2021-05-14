import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import numpy as np

import _init_paths
import utils as utl
from configs.thumos import parse_second_args as parse_args
from models import build_model

def to_device(x, device):
    return x.unsqueeze(0).to(device)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_score_metrics = []
    enc_target_metrics = []

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    softmax = nn.Softmax(dim=1).to(device)

    background_score = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            camera_inputs = np.load(osp.join(args.data_root, args.camera_feature, session+'.npy'), mmap_mode='r')
            motion_inputs = np.load(osp.join(args.data_root, args.motion_feature, session+'.npy'), mmap_mode='r')
            target = np.load(osp.join(args.data_root, 'target', session+'.npy'))
            enc_hx = to_device(torch.zeros(model.hidden_size), device)
            enc_cx = to_device(torch.zeros(model.hidden_size), device)

            for l in range(target.shape[0]):
                if l < args.step_size:
                    enc_score_metrics.append(background_score)
                else:
                    camera_input = to_device(
                        torch.as_tensor(camera_inputs[l-args.step_size].astype(np.float32)), device)
                    motion_input = to_device(
                        torch.as_tensor(motion_inputs[l-args.step_size].astype(np.float32)), device)

                    enc_hx, enc_cx, enc_score = \
                            model.step(camera_input, motion_input, enc_hx, enc_cx)

                    enc_score_metrics.append(softmax(enc_score).cpu().numpy()[0])

                enc_target_metrics.append(target[l])

        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(
            session, session_idx, len(args.test_session_set), end - start))

    save_dir = osp.dirname(args.checkpoint)
    result_file  = osp.basename(args.checkpoint).replace('.pth', '.json')
    # Compute result for encoder

    if args.dataset == "THUMOS":
        utl.compute_result_multilabel(args.dataset, args.class_index,
                                    enc_score_metrics, enc_target_metrics,
                                    save_dir, result_file, ignore_class=[0,21], save=True, verbose=True)
    elif args.dataset == "TVSeries":
        utl.compute_result_multilabel(args.dataset, args.class_index,
                                    enc_score_metrics, enc_target_metrics,
                                    save_dir, result_file, ignore_class=[0], save=True, verbose=True)



if __name__ == '__main__':
    main(parse_args())
