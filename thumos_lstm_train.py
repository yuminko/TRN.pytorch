import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

import _init_paths 
import lib.utils as utl
from configs.thumos import parse_lstm_args as parse_args
from models import build_model

def main(args):
    this_dir = osp.join(osp.dirname(__file__), '.')
    save_dir = osp.join(this_dir, 'tv_lstm_checkpoints_16')
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
    if args.distributed:
        model = nn.DataParallel(model)
    model = model.to(device)

    if args.dataset == 'THUMOS':
        criterion = utl.MultiCrossEntropyLoss(ignore_index=21).to(device)
    elif args.dataset == "TVSeries":
        criterion = utl.MultiCrossEntropyLoss().to(device)

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
        enc_mAP = 0.0

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
                for batch_idx, (camera_inputs, motion_inputs, enc_target) \
                        in enumerate(data_loaders[phase], start=1):
                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)
                    motion_inputs = motion_inputs.to(device)
                    enc_target = enc_target.to(device).view(-1, args.num_classes)

                    enc_score = model(camera_inputs, motion_inputs)
                    enc_loss = criterion(enc_score, enc_target)
                    enc_losses[phase] += enc_loss.item() * batch_size

                    if args.verbose:
                        print('Epoch: {:2} | iteration: {:3} | enc_loss: {:.5f} '.format(
                            epoch, batch_idx, enc_loss.item()
                        ))

                    if training:
                        optimizer.zero_grad()
                        loss = enc_loss 
                        loss.backward()
                        optimizer.step()
                    else:
                        # Prepare metrics for encoder
                        enc_score = softmax(enc_score).cpu().numpy()
                        enc_target = enc_target.cpu().numpy()
                        enc_score_metrics.extend(enc_score)
                        enc_target_metrics.extend(enc_target)

        end = time.time()

        if args.debug:
            result_file = 'LSTM-inputs-{}-epoch-{}.json'.format(args.inputs, epoch)
            # Compute result for encoder
            enc_mAP = utl.compute_result_multilabel(
                args.class_index,
                enc_score_metrics,
                enc_target_metrics,
                save_dir,
                result_file,
                ignore_class=[21],
                save=True,
            )

        # Output result
        logger.lstm_output(epoch, enc_losses, 
                      len(data_loaders['train'].dataset), len(data_loaders['test'].dataset),
                      enc_mAP,  end - start, debug=args.debug)

        # Save model
        checkpoint_file = 'LSTM-inputs-{}-epoch-{}.pth'.format(args.inputs, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(save_dir, checkpoint_file))

if __name__ == '__main__':
    main(parse_args())
