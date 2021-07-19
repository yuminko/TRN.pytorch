import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

import _init_paths 
import lib.utils as utl
from configs.thumos import parse_second_args as parse_args
from models import build_model

# import numpy as np


torch.set_printoptions(precision=4)


def main(args):
    this_dir = osp.join(osp.dirname(__file__), '.')

    ### make directory for each step size
    # '/dataset/volume1/users/yumin/result'
    #'/data/yumin/result'
    save_dir = osp.join('/dataset/volume1/users/yumin/result', 'delta_{}_checkpoints_method{}_noenc_smoothbeta0.5'.format(args.dataset, args.method))

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    

    command = 'python ' + ' '.join(sys.argv)
    logger = utl.setup_logger(osp.join(this_dir, 'lstm_log.txt'), command=command)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))

    model = build_model(args)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.apply(utl.weights_init)
    if args.distributed:            ### !!!
        model = nn.DataParallel(model)
    model = model.to(device)


    if args.dataset == 'THUMOS':
        criterion1 = utl.MultiCrossEntropyLoss_Delta(num_class=args.num_classes, dirichlet=args.dirichlet, ignore_index=21).to(device)
        # criterion2 = nn.MSELoss()
        # criterion2 = nn.L1Loss()
        criterion2 = nn.SmoothL1Loss()
        # criterion2 = nn.HuberLoss()

    elif args.dataset == "TVSeries":
        criterion = utl.MultiCrossEntropyLoss_Delta(num_class=args.num_classes, dirichlet=args.dirichlet).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if osp.isfile(args.checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        args.start_epoch += checkpoint['epoch']

    softmax = nn.Softmax(dim=1).to(device)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        if epoch == 21:
            args.lr = args.lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        data_loaders = {
            phase: utl.build_data_loader(args, phase)
            for phase in args.phases
        }

        enc_losses = {phase: 0.0 for phase in args.phases}
        enc_score_metrics = []
        enc_target_metrics = []
        delta_score_metrics = []
        delta_target_metrics = []
        enc_mAP = 0.0
        delta_mAP = 0.0

        start = time.time()
        for phase in args.phases:
            training = phase=='train'
            if training:
                model.train(True)
            elif not training and args.debug:
                model.train(False)
            else:
                continue

            with torch.set_grad_enabled(training):
                for batch_idx, (camera_inputs, motion_inputs, enc_target, smooth_target) \
                        in enumerate(data_loaders[phase], start=1):
                    
                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)
                    motion_inputs = motion_inputs.to(device)

                    extend_target = enc_target.to(device)
                    enc_target = enc_target.to(device).view(-1, args.num_classes)
                    smooth_target = smooth_target.to(device)
                    oad_score, delta_score = model(camera_inputs, motion_inputs)

                    oad_before = oad_score.clone().detach()
                    oad_before = oad_before[:,1::,:]

                    ## have to make delta target and compute delta loss

                    new_target = smooth_target[:,1::,:] - oad_before
                    # print('***** DELTA TARGET')
                    # print(new_target)
                    # print('***** DELTA SCORE')
                    # print(delta_score[:,1::,:])

                    oad_loss = criterion1(oad_score, extend_target)
                    delta_loss = criterion2(delta_score[:,1::,:], new_target)   # ignore the first
                    # delta_loss = criterion2(delta_score[:,1::,:], extend_target[:,1::,:])   # without labelsmoothing
                    
                    enc_losses[phase] += oad_loss.item() * batch_size

                    if args.verbose:
                        print('Epoch: {:2} | iteration: {:3} | enc_loss: {:.5f} | delta_loss: {:.5f}'.format(
                            epoch, batch_idx, oad_loss.item(), delta_loss.item()*10
                        ))

                    if training:
                        optimizer.zero_grad()
                        loss = oad_loss + delta_loss * 10
                        loss.backward()
                        optimizer.step()
                    else:
                        # Prepare metrics for encoder
                        enc_score = oad_score.cpu().numpy()    ## softmax check
                        enc_target = extend_target.cpu().numpy()
                        enc_score_metrics.extend(enc_score)
                        enc_target_metrics.extend(enc_target)
                        delta_score_c = delta_score[:,1::,:].reshape(-1, args.num_classes)
                        delta = delta_score_c.cpu().numpy()
                        new_target_c = new_target.reshape(-1, args.num_classes)
                        delta_target = new_target_c.cpu().numpy()
                        delta_score_metrics.extend(delta)
                        delta_target_metrics.extend(delta_target)

        end = time.time()

        if args.debug:
            if epoch % 1 == 0:
                result_file = osp.join(this_dir, 'delta-inputs-{}-epoch-{}.json'.format(args.inputs, epoch))
                # Compute result for encoder
                enc_mAP = utl.compute_result_multilabel(
                    args.dataset,
                    args.class_index,
                    enc_score_metrics,
                    enc_target_metrics,
                    save_dir,
                    result_file,
                    ignore_class=[0,21],
                    save=True,
                )

                delta_mAP = utl.compute_result_multilabel(
                    args.dataset,
                    args.class_index,
                    delta_score_metrics,
                    delta_target_metrics,
                    save_dir,
                    result_file,
                    ignore_class=[0,21],
                    save=True,
                    smooth=True,
                )

        # Output result
        logger.delta_output(epoch, enc_losses, 
                    len(data_loaders['train'].dataset), len(data_loaders['test'].dataset),
                    enc_mAP, delta_mAP, end - start, debug=args.debug)

        # Save model
        checkpoint_file = 'delta-inputs-{}-epoch-{}.pth'.format(args.inputs, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(save_dir, checkpoint_file))

if __name__ == '__main__':
    main(parse_args())
