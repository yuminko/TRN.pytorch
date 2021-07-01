import torch
import torch.nn as nn

__all__ = ['MultiCrossEntropyLoss', 'MultiCrossEntropyLoss_Second', 'MultiCrossEntropyLoss_Delta']

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
 
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), 1)

            return torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)

class MultiCrossEntropyLoss_Second(nn.Module):
    def __init__(self, num_class, dirichlet, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss_Second, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

        self.num_class = num_class

        self.dirichlet = dirichlet
        
    def forward(self, input, target, step_size):

        step = int(step_size)
        num_class = self.num_class

        # print(target.shape)   # [32, 64, 22]
        # print(input.shape)    # [16, 64, 22]    (B, T, C) B 16???

        step_target = []
        step_input = [] 

        step_target.append(target[:, step::, :])
        step_input.append(input[:, step::, :])

        step_target = torch.stack(step_target, dim=1).view(-1,num_class)
        step_input = torch.stack(step_input, dim=1).view(-1, num_class)

        if not self.dirichlet:
            logsoftmax = nn.LogSoftmax(dim=1).to(step_input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]

            if self.dirichlet:
                # print(step_target.shape)  [2048, 22]
                # print(step_input.shape)   [1024, 22]
                output = torch.sum(-step_target[:,notice_index] * torch.log(step_input[:,notice_index]),1)

            else:
                output = torch.sum(-step_target[:,notice_index] * logsoftmax(step_input[:,notice_index]),1)

            return torch.mean(output[step_target[:,self.ignore_index] != 1])

        else:
            if self.dirichlet:
                output = torch.sum(-step_target * torch.log(step_input), 1)
            else:
                output = torch.sum(-step_target * logsoftmax(step_input), 1)
                
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)

class MultiCrossEntropyLoss_Delta(nn.Module):
    def __init__(self, num_class, dirichlet, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss_Delta, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

        self.num_class = num_class

        self.dirichlet = dirichlet
        
    def forward(self, input, target):

        target = target.view(-1, self.num_class)
        input = input.view(-1, self.num_class)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]

            if self.dirichlet:
                output = torch.sum(-target[:,notice_index] * torch.log(input[:,notice_index]),1)

            else:
                logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
                output = torch.sum(-target[:,notice_index] * logsoftmax(input[:,notice_index]),1)

            return torch.mean(output[target[:,self.ignore_index] != 1])

        else:
            if self.dirichlet:
                output = torch.sum(-target * torch.log(input), 1)
            else:
                logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
                output = torch.sum(-target * logsoftmax(input), 1)
                
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)
