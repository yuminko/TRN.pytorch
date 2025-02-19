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

np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})


def to_device(x, device):
    return x.unsqueeze(0).to(device)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_score_metrics = []
    delta_score_metrics = []

    state_metrics = []
    final_state = []

    enc_target_metrics = []

    enc_variance_score_metrics = []
    delta_variance_score_metrics = []

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))

    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)


    softmax = nn.Softmax(dim=1).to(device)

    thumos_background_score = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    tvseries_background_score = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0])


    for session_idx, session in enumerate(args.test_session_set, start=1):

        start = time.time()

        with torch.set_grad_enabled(False):
            camera_inputs = np.load(osp.join(args.data_root, args.camera_feature, session+'.npy'), mmap_mode='r')
            motion_inputs = np.load(osp.join(args.data_root, args.motion_feature, session+'.npy'), mmap_mode='r')
            
            target = np.load(osp.join(args.data_root, 'target', session+'.npy'))

            enc_hx = to_device(torch.zeros(model.hidden_size), device)
            enc_cx = to_device(torch.zeros(model.hidden_size), device)
            d_enc_hx = to_device(torch.zeros(model.hidden_size), device)
            d_enc_cx = to_device(torch.zeros(model.hidden_size), device)

            dummy_score = to_device(torch.zeros(args.num_classes), device)
            oad_score = []
            oad_score.append(dummy_score)



            for l in range(target.shape[0]):
                # print(l)

                enc_target_metrics.append(target[l])

                delta_score_metrics.append(thumos_background_score)

                camera_input = to_device(
                    torch.as_tensor(camera_inputs[l].astype(np.float32)), device)
                motion_input = to_device(
                    torch.as_tensor(motion_inputs[l].astype(np.float32)), device)

                delta_camera_input = to_device(
                    torch.as_tensor(camera_inputs[l-1].astype(np.float32)), device)
                delta_motion_input = to_device(
                    torch.as_tensor(motion_inputs[l-1].astype(np.float32)), device)


                enc_hx, enc_cx, enc_score, enc_var = \
                    model.step(camera_input, motion_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, dummy_score, delta=False)
                # enc_hx, enc_cx, enc_score, enc_var = \
                #     model.step(camera_input, motion_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, oad_score[-1], delta=False)
                
                oad_score.append(enc_score)
                
                d_enc_hx, d_enc_cx, delta_score, delta_var = \
                    model.step(delta_camera_input, delta_motion_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, oad_score[-2], delta=True)

                if args.dirichlet:
                    enc_score_metrics.append(enc_score.cpu().numpy()[0])

                else:
                    enc_score_metrics.append(softmax(enc_score).cpu().numpy()[0])

                delta_score_metrics.append(delta_score.cpu().numpy()[0])

                if l > 0:
                    state = np.add(state_metrics[-1], delta_score.view(-1,1).cpu())

                else:
                    state = enc_score


                ## compute coefficient of kalman filter
                
                inverse_enc = np.linalg.inv(enc_var)
                inverse_delta = np.linalg.inv(delta_var)
                # inverse_delta = np.linalg.inv(delta_var.cpu())
                summ = np.add(inverse_enc, inverse_delta)
                inverse_summ = np.linalg.inv(summ)
                oad_coeff = np.dot(inverse_summ, inverse_enc)
                delta_coeff = np.dot(inverse_summ, inverse_delta)
                # print(np.add(oad_coeff,delta_coeff))  # check identity matrix
                oad_update = np.dot(oad_coeff, enc_score.view(-1,1).cpu())
                delta_update = np.dot(delta_coeff, state.view(-1,1).cpu())
                state_update = np.add(oad_update, delta_update)
                state_metrics.append(state_update)
                final_state.append(state_update.reshape((22,)))

                # print('##########   oad_coeff')
                # print(np.diag(oad_coeff))
                # print('##########   delta_coeff')
                # print(np.diag(delta_coeff))



                # if np.where(target[l]==1)[0][0] != 0:

                #     print(l)

                #     asdf

                # if np.amax(state.cpu().numpy()) > 2:

        #         if np.where(enc_score.cpu().numpy()[0]==np.amax(enc_score.cpu().numpy()[0]))[0][0] == np.where(target[l]==1)[0][0]:
        #             if np.where(state_update.reshape((22,))==np.amax(state_update.reshape((22,))))[0][0] != np.where(target[l]==1)[0][0]:
    
        #                 print('##########   target')
        #                 print(target[l])
        #                 # print('##########   enc_var')
        #                 # print(enc_var)
        #                 # print('##########   inverse_enc_var')
        #                 # print(inverse_enc)
        #                 # print('##########   delta_var')
        #                 # print(delta_var.cpu().numpy())
        #                 # print('##########   inverse_delta_var')
        #                 # print(inverse_delta)
        #                 # print('##########   sum')
        #                 # print(summ)
        #                 # print('##########   inverse_sum')
        #                 # print(inverse_summ)
                            # print('##########   oad_coeff')
                            # print(oad_coeff)
                            # print('##########   delta_coeff')
                            # print(delta_coeff)
        #                 print('##########   oad_score')
        #                 print(enc_score.cpu().numpy()[0])
        #                 print('##########   delta_score')
        #                 print(delta_score.cpu().numpy()[0])
        #                 print('##########   state')
        #                 print(state.cpu().numpy())
        #                 print('##########   state before')
        #                 print(state_metrics[-2])
        #                 print('##########   oad_update')
        #                 print(oad_update)
        #                 print('##########   delta_update')
        #                 print(delta_update)
        #                 print('##########   state_update')
        #                 print(state_update)
        #                 print(state_update.reshape((22,)))

        #                 if np.amax(state.cpu().numpy()) > 2:
        #                     asdf

        # asdf


            
        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(
            session, session_idx, len(args.test_session_set), end - start))

    save_dir = osp.dirname(args.checkpoint)
    result_file  = osp.basename(args.checkpoint).replace('.pth', '.json')
    # Compute result for encoder

    if args.dataset == "THUMOS":        
        print(len(state_metrics))
        print(len(enc_target_metrics))
        utl.compute_result_multilabel(args.dataset, args.class_index,
                                    final_state, enc_target_metrics,
                                    save_dir, result_file, ignore_class=[0,21], save=True, verbose=True)
        print('oad mAP')
        utl.compute_result_multilabel(args.dataset, args.class_index,
                                        enc_score_metrics, enc_target_metrics,
                                        save_dir, result_file, ignore_class=[0,21], save=True, verbose=True)

    elif args.dataset == "TVSeries":
        for i, steps in enumerate(args.step_size):
            print('Step size:   ', steps)
            utl.compute_result_multilabel(args.dataset, args.class_index,
                                    enc_score_metrics[i], enc_target_metrics,
                                    save_dir, result_file, ignore_class=[0], save=True, verbose=True)



if __name__ == '__main__':
    main(parse_args())
