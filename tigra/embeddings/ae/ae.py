from tigra.embeddings.ae.base import SymetricAEBase
from tigra.embeddings.ae.common import MLP, BatchGenerator, BatchPredictor
from tigra.embeddings.embedding import SingletonEmbedding

from torch import nn
import torch
from typing import Sequence


class AE_Model(nn.Module):
    encoder: MLP
    decoder: MLP

    def __init__(
        self,
        input_size: int,
        output_size: int,
        look_back: int,
        layer_sizes: Sequence[int],
        bias=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder = MLP(input_size * look_back, output_size, layer_sizes, bias)
        self.decoder = MLP(output_size, input_size, layer_sizes[::-1], bias)

    def forward(self, x):
        hx = self.encoder(x)
        x_pred = self.decoder(hx)
        return hx, x_pred


class DynAE(SymetricAEBase):
    def __init__(
        self,
        dim,
        look_back,
        beta,
        nu1,
        nu2,
        batch_size,
        epochs,
        learning_rate,
        weight_decay,
        n_units,
        last_only=True,
        bias=True,
        shuffle=True,
        has_cuda=False,
        verbose=0,
        **kawrgs,
    ):
        super().__init__(
            dim,
            look_back,
            beta,
            nu1,
            nu2,
            batch_size,
            epochs,
            learning_rate,
            weight_decay,
            n_units,
            last_only,
            bias,
            shuffle,
            has_cuda,
            verbose,
            **kawrgs,
        )
        self.name = "DynAE"

    def __construct(self, data, prediction_index, num_nodes, node_list):
        model = AE_Model(
            input_size=num_nodes,
            output_size=self.dim,
            look_back=self.look_back,
            layer_sizes=self.n_units,
            bias=self.bias,
        )
        # TODO: Implement loss base class
        loss_model = None
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        batch_generator = BatchGenerator(
            node_list=node_list,
            batch_size=self.batch_size,
            look_back=self.look_back,
            beta=self.beta,
            shuffle=self.shuffle,
            has_cuda=self.has_cuda,
        )
        batch_predictor = BatchPredictor(
            node_list=node_list,
            batch_size=self.batch_size,
            has_cuda=self.has_cuda,
        )

        train_size = data.get_num_snaps() - self.look_back
        element_num = num_nodes * train_size
        batch_num = element_num // self.batch_size
        if element_num % self.batch_size != 0:
            batch_num += 1
        graphs = data.get_scipy_sparse_matrices()

        for i in range(self.epochs):
            for j in range(batch_num):
                generator = batch_generator.generate(graphs)
                x_pre_batches, x_cur_batch, y_batch = next(generator)
                x_pre_batches = x_pre_batches.reshape(self.batch_size, -1)
                _, x_pre_batch = model(x_pre_batches)
                loss_input_list = [x_pre_batch, x_cur_batch, y_batch]
                loss = loss_model(model, loss_input_list)
                loss.backward()

                if j == batch_num - 1:
                    optimizer.step()
                    model.zero_grad()

        embedding_mat, _ = batch_predictor.predict(model, graphs[train_size:])
        assert embedding_mat != 0

        singleton = SingletonEmbedding()

        # SVC, 27.10.2024
        learn_graph = data.get_matrix_as_graph(prediction_index)
        node_list = learn_graph.G.nodes()

        for node in node_list:
            singleton.add_embedding_vector(node, embedding_mat[node].tolist())

        return singleton


__all__ = ["DynAE"]
