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
        # print(input.shape)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            # print(input[:, notice_index].shape)
            # print(target[:, notice_index].shape)
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), 1)

            # print(output[target[:, self.ignore_index] != 1])
            # print(output[target[:, self.ignore_index] != 1].shape)
            # print([target[:, self.ignore_index] != 1])

            return torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)

class MultiCrossEntropyLoss_Second(nn.Module):
    def __init__(self, batch_size, step_size, num_class, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss_Second, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

        self.size = batch_size
        self.step_size = step_size
        self.num_class = num_class
        
    def forward(self, input, target):

        size = self.size
        step = self.step_size
        num_class = self.num_class

        step_target = []
        step_input = []

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]

            # print(target.shape)
            # print(input.shape)

            for i in range(target.shape[0]):
                step_target.append(target[i, step::, :])
                step_input.append(input[i, step::, :])


            step_target = torch.stack(step_target, dim=1).view(-1,num_class)
            step_input = torch.stack(step_input, dim=1).view(-1, num_class)

            logsoftmax = nn.LogSoftmax(dim=1).to(step_input.device)

            output = torch.sum(-step_target[:,notice_index] * logsoftmax(step_input[:,notice_index]),1)

            return torch.mean(output[step_target[:,self.ignore_index] != 1])

    # def forward(self, input, target):
    #     logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

    #     if self.ignore_index >= 0:
    #         notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]

    #         step = self.step_size

    #         output = -target[:, step:: , notice_index] * logsoftmax(input[:, step::, notice_index])
    #         output = output.view(-1, len(notice_index))
    #         output = torch.sum(output, 1)

    #         # output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), 1)

    #         index = target[:, step::, self.ignore_index] != 1
    #         index = index.reshape(-1)

    #         return torch.mean(output[index])
    #     else:
    #         output = torch.sum(-target * logsoftmax(input), 1)
    #         if self.size_average:
    #             return torch.mean(output)
    #         else:
    #             return torch.sum(output)