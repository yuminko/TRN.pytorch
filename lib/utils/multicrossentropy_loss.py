import torch
import torch.nn as nn

__all__ = ['MultiCrossEntropyLoss', 'MultiCrossEntropyLoss_Second']

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        print(input.shape)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            # print(input[:, notice_index].shape)
            # print(target[:, notice_index].shape)
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), 1)
            return torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)

class MultiCrossEntropyLoss_Second(nn.Module):
    def __init__(self, enc_steps, step_size,  size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss_Second, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

        self.size = enc_steps / 2
        self.step_size = step_size
        

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            start = int(self.step_size * self.size)

            print(start)
            print(input[start::, notice_index].shape)
            
            output = torch.sum(-target[start::, notice_index] * logsoftmax(input[start::, notice_index]), 1)
            return torch.mean(output[target[start::, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)