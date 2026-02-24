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
    regularization: RegularizationLoss

    def __init__(self, nu1, nu2):
        super(DynGraph2VecLoss, self).__init__()
        self.regularization = RegularizationLoss(nu1, nu2)

    def forward(self, model, input_list):
        x_reconstruct, x_real, y_penalty = input_list[0], input_list[1], input_list[2]
        assert len(input_list) >= 3
        reconstruct_loss = torch.mean(torch.sum(torch.square((x_reconstruct - x_real) * y_penalty), dim=1))
        regularization_loss = self.regularization(model)
        # print('total loss: ', main_loss.item(), ', reconst loss: ', reconstruct_loss.item(), ', L1 loss: ', l1_loss.item(), ', L2 loss: ', l2_loss.item())
        return reconstruct_loss + regularization_loss

class GraphLIDLoss(nn.Module):
    """
    Loss function for LID

    Compares embedding complexity (L2 norm) with node LID values.
    """

    def __init__(self):
        super(GraphLIDLoss, self).__init__()

    def forward(self, model, input_list):
        # model parameter kept for interface compatibility but not used
        # input_list: [x_pred_batch, x_cur_batch, y_batch, embedding_batch]
        if len(input_list) < 4:
            raise ValueError("GraphLIDLoss requires embeddings in input_list[3]")

        _, _, lid_batch, Z = input_list

        z_centered = Z - Z.mean(dim=0, keepdim=True)
        latent_complexity = torch.norm(z_centered, dim=1)

        latent_complexity = (
            latent_complexity - latent_complexity.mean()
        ) / (latent_complexity.std() + 1e-8)

        # lid_batch has shape (batch_size, node_num) with same value across each row
        # Extract just the first column to get shape (batch_size,)
        lid_values = lid_batch[:, 0]

        return torch.mean(torch.abs(latent_complexity - lid_values))

class HUBLoss(nn.Module):
    """
    Loss function for HUB
    """

    def __init__(self):
        super(HUBLoss, self).__init__()

    def forward(self, model, input_list):
        x_reconstruct, x_real = input_list[0], input_list[1]
        # input_list[2] contains penalty tensor but HUBLoss doesn't use it
        assert len(input_list) >= 2
        loss = torch.mean(torch.sum(torch.abs(x_reconstruct - x_real), dim=1))
        return loss


class CombinedDynGraphLoss(nn.Module):
    """
    Combined loss: reconstruction + LID regularization

    This loss combines:
    1. DynGraph2VecLoss (reconstruction with penalty weighting)
    2. GraphLIDLoss (embedding complexity matching LID)

    Total loss = reconstruction_loss + lambda_lid * lid_loss
    """

    reconstruction_loss: DynGraph2VecLoss
    lid_loss: GraphLIDLoss
    lambda_lid: float

    def __init__(self, nu1: float, nu2: float, lambda_lid: float = 0.1):
        """
        Parameters:
        -----------
        nu1 : float
            L1 regularization weight
        nu2 : float
            L2 regularization weight
        lambda_lid : float
            Weight for LID loss component (default: 0.1)
        """
        super(CombinedDynGraphLoss, self).__init__()
        self.reconstruction_loss = DynGraph2VecLoss(nu1, nu2)
        self.lid_loss = GraphLIDLoss()
        self.lambda_lid = lambda_lid

    def forward(self, model, input_list):
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss(model, input_list)

        # Compute LID loss
        lid_loss_value = self.lid_loss(model, input_list)

        # Combine losses
        total_loss = recon_loss + self.lambda_lid * lid_loss_value

        return total_loss
