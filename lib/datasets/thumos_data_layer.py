import os.path as osp

import torch
import torch.utils.data as data
import numpy as np
import math

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


def gaussian_kernel(k_size, sigma): 

    arr = np.arange(math.trunc(k_size/2)*(-1), math.ceil(k_size/2)+1,1)
    kernel_raw = np.exp((-arr*arr)/(2*sigma*sigma))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


class LSTMTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.motion_feature = args.motion_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.training = phase=='train'

        self.inputs = []

        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            seed = np.random.randint(self.enc_steps) if self.training else 0
            smooth = target.copy()

            ### smooth the original label

            # for i in range(len(target[0])):
            #     column = target[:,i]
            #     stack = []
            #     for j in range(len(column)):
            #         if column[j] == 1:
            #             smooth[j][i] = 1
            #         else:
            #             n = window_size // 2
            #             if  j <= len(column)-1-n:
            #                 stack_part = stack[-n:]
            #                 summ = sum(stack_part)   
            #                 for m in range(n+1):
            #                     summ += column[j+m]
            #                 new = summ / (len(stack_part) + 1 + n)
            #                 smooth[j][i] = new
            #             else:
            #                 stack_part = stack[-n:]
            #                 summ = sum(stack_part)  
            #                 count = 0
            #                 while j+count+1 != len(column):
            #                     summ += column[j+count]
            #                     count += 1
            #                 new = summ / (len(stack) + count)
            #                 smooth[j][i] = new
            #         stack.append(column[j])


            for i in range(len(target[0])):
                column = target[:,i]
                kernel = gaussian_kernel(5,3)
                new = np.convolve(column, kernel ,'same')
                smooth[:,i] = new
                for j in range(len(column)):
                    if column[j] == 1:
                        smooth[j][i] = 1
 

           
            # y1 = target[:,1][:100]
            # y2 = smooth[:,1][:100]
            # x = np.linspace(0,len(y1),len(y1))
            # plt.plot(x,y1, color='red')
            # plt.plot(x,y2, color='green')
            # plt.savefig('/dataset/NAS2/CIPLAB/users/ymko/TRN.pytorch/windowsize20_gauss.png')
            # asdf
            

            ### regularize the smooth target
            for r in range(len(smooth[0])):
                row = smooth[r]
                summ = sum(row)
                if summ != 1:
                    # print(row)
                    row = row / summ
                    # print(row)

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



