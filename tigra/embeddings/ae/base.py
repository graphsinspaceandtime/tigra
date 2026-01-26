from abc import abstractmethod

import torch
from torch import nn
from torch.autograd import Variable


class BaseLoss(nn.Module):
    """
    Base class for auto encoder loss functions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, model, input_list):
        pass

# L1 and L2 regularization loss
class RegularizationLoss(nn.Module):
    nu1: float
    nu2: float

    def __init__(self, nu1, nu2):
        super(RegularizationLoss, self).__init__()
        self.nu1 = nu1
        self.nu2 = nu2

    @staticmethod
    def get_weight(model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                # print('name: ', name)
                weight_list.append(weight)
        return weight_list

    def forward(self, model):
        loss = Variable(torch.FloatTensor([0.]), requires_grad=True).cuda() if torch.cuda.is_available() else Variable(torch.FloatTensor([0.]), requires_grad=True)
        # No L1 regularization and no L2 regularization
        if self.nu1 == 0. and self.nu2 == 0.:
            return loss
        # calculate L1-regularization loss and L2-regularization loss
        weight_list = self.get_weight(model)
        weight_num = len(weight_list)
        # print('weight num', weight_num)
        l1_reg_loss, l2_reg_loss = 0, 0
        for name, weight in weight_list:
            if self.nu1 > 0:
                l1_reg = torch.norm(weight, p=1)
                l1_reg_loss = l1_reg_loss + l1_reg
            if self.nu2 > 0:
                l2_reg = torch.norm(weight, p=2)
                l2_reg_loss = l2_reg_loss + l2_reg
        l1_loss = self.nu1 * l1_reg_loss / weight_num
        l2_loss = self.nu2 * l2_reg_loss / weight_num
        return l1_loss + l2_loss

# Loss used for DynAE, DynRNN, DynAERNN
class DynGraph2VecLoss(nn.Module):
    beta: float
    regularization: RegularizationLoss

    def __init__(self, beta, nu1, nu2):
        super(DynGraph2VecLoss, self).__init__()
        self.beta = beta
        self.regularization = RegularizationLoss(nu1, nu2)

    def forward(self, model, input_list):
        x_reconstruct, x_real, y_penalty = input_list[0], input_list[1], input_list[2]
        assert len(input_list) == 3
        reconstruct_loss = torch.mean(torch.sum(torch.square((x_reconstruct - x_real) * y_penalty), dim=1))
        regularization_loss = self.regularization(model)
        # print('total loss: ', main_loss.item(), ', reconst loss: ', reconstruct_loss.item(), ', L1 loss: ', l1_loss.item(), ', L2 loss: ', l2_loss.item())
        return reconstruct_loss + regularization_loss

class GraphLIDLoss(nn.Module):
    """
    Loss function for LID
    """

    def __init__(self):
        super(GraphLIDLoss, self).__init__()

    def forward(self, model, input_list):
        _, _, lid_batch = input_list
        Z = model.z

        z_centered = Z - Z.mean(dim=0, keepdim=True)
        latent_complexity = torch.norm(z_centered, dim=1)

        latent_complexity = (
            latent_complexity - latent_complexity.mean()
        ) / (latent_complexity.std() + 1e-8)

        return torch.mean(torch.abs(latent_complexity - lid_batch))

class HUBLoss(nn.Module):
    """
    Loss function for HUB
    """

    def __init__(self):
        super(HUBLoss, self).__init__()

    def forward(self, model, input_list):
        x_reconstruct, x_real = input_list[0], input_list[1]
        assert len(input_list) == 2
        loss = torch.mean(torch.sum(torch.abs(x_reconstruct - x_real), dim=1))
        return loss
