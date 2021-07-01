import os.path as osp

import torch
import torch.utils.data as data
import numpy as np

import matplotlib.pyplot as plt

class TRNTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.motion_feature = args.motion_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.training = phase=='train'

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, self.enc_steps)):
                enc_target = target[start:end]
                dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                self.inputs.append([
                    session, start, end, enc_target, dec_target,
                ])

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i,j] = target_vector[i+j,:]
        return target_matrix

    def __getitem__(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]

        camera_inputs = np.load(
            osp.join(self.data_root, self.camera_feature, session+'.npy'), mmap_mode='r')[start:end]
        camera_inputs = torch.as_tensor(camera_inputs.astype(np.float32))
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session+'.npy'), mmap_mode='r')[start:end]
        motion_inputs = torch.as_tensor(motion_inputs.astype(np.float32))
        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        dec_target = torch.as_tensor(dec_target.astype(np.float32))

        return camera_inputs, motion_inputs, enc_target, dec_target.view(-1, enc_target.shape[-1])

    def __len__(self):
        return len(self.inputs)

class LSTMTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.motion_feature = args.motion_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.training = phase=='train'

        self.inputs = []

        window_size = 9

        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            seed = np.random.randint(self.enc_steps) if self.training else 0
            smooth = target.copy()

            ### smooth the original label

            for i in range(len(target[0])):
                column = target[:,i]
                stack = []

                for j in range(len(column)):
                    stack.append(column[j])
                    stack = stack[-window_size:]
                    new = sum(stack) / len(stack)
                    smooth[j][i] = new

            ### regularize the smooth target
            # for r in range(len(target[0])):
            #     row = target[r]
            #     summ = sum(row)
            #     if summ != 1:
            #         print(row)

            # asdf

            ### 
   
            for start, end in zip(
                range(seed, target.shape[0], self.enc_steps),
                range(seed + self.enc_steps, target.shape[0], self.enc_steps)):
                enc_target = target[start:end]
                smooth_target = smooth[start:end]
                self.inputs.append([
                    session, start, end, enc_target, smooth_target
                ])


    def __getitem__(self, index):
        session, start, end, enc_target, smooth_target = self.inputs[index]

        camera_inputs = np.load(
            osp.join(self.data_root, self.camera_feature, session+'.npy'), mmap_mode='r')[start:end]
        camera_inputs = torch.as_tensor(camera_inputs.astype(np.float32))
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session+'.npy'), mmap_mode='r')[start:end]
        motion_inputs = torch.as_tensor(motion_inputs.astype(np.float32))
        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        smooth_target = torch.as_tensor(smooth_target.astype(np.float32))


        return camera_inputs, motion_inputs, enc_target, smooth_target

    def __len__(self):
        return len(self.inputs)

