import tensorflow as tf
from keras import losses
from tensorflow.python.keras.utils.losses_utils import ReductionV2

import numpy as np
import torch
from torch import nn, autograd as ag


class WeightedMSEX(losses.Loss):
    def __init__(self, num_nodes, reduction=ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self._num_nodes = num_nodes

    def call(self, y_true, y_pred):
        """Hack: This fn doesn't accept additional arguments.
                We use y_true to pass them.
        y_pred: Contains x_hat - x_pred
        y_true: Contains b
        """
        return tf.reduce_sum(
            tf.square(y_pred * y_true[:, 0 : self._num_nodes]), axis=-1
        )


class BatchGenerator:
    node_list: list
    node_num: int
    batch_size: int
    look_back: int
    beta: float
    shuffle: bool
    has_cuda: bool

    def __init__(
        self, node_list, batch_size, look_back, beta, shuffle=True, has_cuda=False
    ):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.look_back = look_back
        self.beta = beta
        self.shuffle = shuffle
        self.has_cuda = has_cuda

    def generate(self, graph_list):
        graph_num = len(graph_list)
        train_size = graph_num - self.look_back
        assert train_size > 0
        all_node_num = self.node_num * train_size
        batch_num = all_node_num // self.batch_size
        if all_node_num % self.batch_size != 0:
            batch_num += 1
        node_indices = np.arange(all_node_num)

        if self.shuffle:
            np.random.shuffle(node_indices)
        counter = 0
        while True:
            batch_indices = node_indices[
                self.batch_size
                * counter : min(all_node_num, self.batch_size * (counter + 1))
            ]
            x_pre_batch = torch.zeros((self.batch_size, self.look_back, self.node_num))
            x_pre_batch = x_pre_batch.cuda() if self.has_cuda else x_pre_batch
            x_cur_batch = torch.zeros(
                (self.batch_size, self.node_num), device=x_pre_batch.device
            )
            y_batch = torch.ones(
                x_cur_batch.shape, device=x_pre_batch.device
            )  # penalty tensor for x_cur_batch

            for idx, record_id in enumerate(batch_indices):
                graph_idx = record_id // self.node_num
                node_idx = record_id % self.node_num
                for step in range(self.look_back):
                    # graph is a scipy.sparse.lil_matrix
                    pre_tensor = torch.tensor(
                        graph_list[graph_idx + step][node_idx, :].toarray(),
                        device=x_pre_batch.device,
                    )
                    x_pre_batch[idx, step, :] = pre_tensor
                # graph is a scipy.sparse.lil_matrix
                cur_tensor = torch.tensor(
                    graph_list[graph_idx + self.look_back][node_idx, :].toarray(),
                    device=x_pre_batch.device,
                )
                x_cur_batch[idx] = cur_tensor

            y_batch[x_cur_batch != 0] = self.beta
            counter += 1
            yield x_pre_batch, x_cur_batch, y_batch

            if counter == batch_num:
                if self.shuffle:
                    np.random.shuffle(node_indices)
                counter = 0


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
            if "weight" in name:
                weight = (name, param)
                # print('name: ', name)
                weight_list.append(weight)
        return weight_list

    def forward(self, model):
        loss = (
            ag.Variable(torch.FloatTensor([0.0]), requires_grad=True).cuda()
            if torch.cuda.is_available()
            else ag.Variable(torch.FloatTensor([0.0]), requires_grad=True)
        )
        # No L1 regularization and no L2 regularization
        if self.nu1 == 0.0 and self.nu2 == 0.0:
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
        reconstruct_loss = torch.mean(
            torch.sum(torch.square((x_reconstruct - x_real) * y_penalty), dim=1)
        )
        regularization_loss = self.regularization(model)
        # print('total loss: ', main_loss.item(), ', reconst loss: ', reconstruct_loss.item(), ', L1 loss: ', l1_loss.item(), ', L2 loss: ', l2_loss.item())
        return reconstruct_loss + regularization_loss


class BatchPredictor:
    node_list: list
    node_num: int
    batch_size: int
    has_cuda: bool

    def __init__(self, node_list, batch_size, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.has_cuda = has_cuda

    def get_predict_res(
        self,
        graph_list,
        model,
        batch_indices,
        counter,
        look_back,
        embedding_mat,
        x_pred,
    ):
        batch_size = len(batch_indices)
        x_pre_batches = torch.zeros((batch_size, look_back, self.node_num))
        x_pre_batches = x_pre_batches.cuda() if self.has_cuda else x_pre_batches

        for idx, node_idx in enumerate(batch_indices):
            for step in range(look_back):
                # graph is a scipy.sparse.lil_matrix
                pre_tensor = torch.tensor(
                    graph_list[step][node_idx, :].toarray(), device=x_pre_batches.device
                )
                x_pre_batches[idx, step, :] = pre_tensor
        # DynAE uses 2D tensor as its input
        if model.method_name == "DynAE":
            x_pre_batches = x_pre_batches.reshape(batch_size, -1)
        embedding_mat_batch, x_pred_batch = model(x_pre_batches)
        if counter:
            embedding_mat = torch.cat((embedding_mat, embedding_mat_batch), dim=0)
            x_pred = torch.cat((x_pred, x_pred_batch), dim=0)
        else:
            embedding_mat = embedding_mat_batch
            x_pred = x_pred_batch
        return embedding_mat, x_pred

    def predict(self, model, graph_list):
        look_back = len(graph_list)
        counter = 0
        embedding_mat, x_pred = 0, 0
        batch_num = self.node_num // self.batch_size

        while counter < batch_num:
            batch_indices = range(
                self.batch_size * counter, self.batch_size * (counter + 1)
            )
            embedding_mat, x_pred = self.get_predict_res(
                graph_list,
                model,
                batch_indices,
                counter,
                look_back,
                embedding_mat,
                x_pred,
            )
            counter += 1
        # has a remaining batch
        if self.node_num % self.batch_size != 0:
            remain_indices = range(self.batch_size * counter, self.node_num)
            embedding_mat, x_pred = self.get_predict_res(
                graph_list,
                model,
                remain_indices,
                counter,
                look_back,
                embedding_mat,
                x_pred,
            )
        return embedding_mat, x_pred
