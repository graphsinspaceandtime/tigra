from typing import List

from tigra.embeddings.sota.dynae.ae_embedding_model import (
    DynamicAutoEncoderEmbeddingModel,
    EmbedderModel,
)
from tigra.embeddings.sota.dynae.dnn_utils import WeightedMSEX
from keras import layers, regularizers, models, optimizers, Sequential
import numpy as np


import torch
from torch import nn, autograd as ag
from torch.nn import functional as F


class DynAE(DynamicAutoEncoderEmbeddingModel):
    def __init__(self, d, **kwargs) -> None:
        """Initialize a DynAE embedder.

        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            n_prev_graphs: lookback for previous graphs
            n_units: vector of length K-1 containing #units in hidden
                     layers of encoder/decoder, not including the units
                     in the embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of iterations for embedding
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD or Adam
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
            savefilesuffix: suffix for saving the files
        """
        super().__init__(d, method_name="DynAE", **kwargs)

    def learn_embedding(self, graphs: list):
        num_nodes = graphs[0].shape[0]
        # TODO: Time?
        # TODO: Config?
        num_iter = self._n_iter
        # encoder = self.__get_encoder(
        #     num_nodes, self._d, self._n_units, self._nu1, self._nu2, self._actfn
        # )
        # decoder = self.__get_decoder(
        #     num_nodes, self._d, self._n_units, self._nu1, self._nu2, self._actfn
        # )
        # autoencoder = self.__get_autoencoder(encoder, decoder, (num_nodes, 1))
        input_layer, encoder_output = self.__get_our_encoder(
            num_nodes, self._d, self._n_units, self._nu1, self._nu2, self._actfn
        )
        decoder_output = self.__get_our_decoder(
            num_nodes,
            self._d,
            self._n_units,
            self._nu1,
            self._nu2,
            self._actfn,
            encoder_output,
        )
        autoencoder = self.__get_our_autoencoder(
            input_layer, encoder_output, decoder_output
        )

        x_in = layers.Input(shape=(num_nodes,), name="x_in")
        x_pred = layers.Input(shape=(num_nodes,), name="x_pred")
        [x_hat, y] = autoencoder(x_in)
        x_diff = layers.Subtract()([x_hat, x_in])

        # def weighted_mse_x(y_true, y_pred):
        #     """Hack: This fn doesn't accept additional arguments.
        #           We use y_true to pass them.
        #     y_pred: Contains x_hat - x_pred
        #     y_true: Contains b
        #     """
        #     return tf.reduce_sum(tf.square(y_pred * y_true[:, 0:num_nodes]), axis=-1)

        model = models.Model(inputs=[x_in, x_pred], outputs=x_diff)
        # TODO: Weight decay no longer exists
        sgd = optimizers.SGD(
            learning_rate=self._xeta,
            # weight_decay=1e-5,
            momentum=0.99,
            nesterov=True,
        )
        adam = optimizers.Adam(
            learning_rate=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )
        # Using WeightedMSEX since it's a modern way to implement a loss function
        # weighted_mse_x could also be used from the original implementation
        model.compile(optimizer=sgd, loss=WeightedMSEX(num_nodes))
        model.summary()

        history = model.fit(
            self.__batch_generator_dynae(
                graphs, self._beta, self._n_batch, self._n_prev_graphs, True
            ),
            epochs=self._n_iter,
            steps_per_epoch=(num_nodes * self._n_prev_graphs) // self._n_batch,
            verbose=1,
        )

        loss = history.history["loss"]

        # TODO: Do we want to do this?
        if loss[0] == np.inf or np.isnan(loss[0]):
            print("Model diverged. Assigning random embeddings")
            self._Y = np.random.randn(self._node_num, self._d)
        else:
            self._Y, self._next_adj = self.__model_batch_predictor_dynae(
                autoencoder,
                graphs[len(graphs) - self._n_prev_graphs :],
                num_nodes,
                self._n_batch,
            )

    def get_embedding(self) -> List:
        return self._Y

    def __get_encoder(self, num_nodes, d, n_units, nu1, nu2, activation_function):
        K = len(n_units) + 1

        # input
        x = layers.Input(shape=(num_nodes, 1))

        # encoder layers
        y = [None] * (K + 1)
        y[0] = x
        for i in range(K - 1):
            y[i + 1] = layers.Dense(
                n_units[i],
                activation=activation_function,
                bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            )(y[i])
        y[K] = layers.Dense(
            d,
            activation=activation_function,
            bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
        )(y[K - 1])
        # TODO: Is the next line right?
        # encoder = models.Model(inputs=x, outputs=y[K])
        encoder = models.Model(inputs=x, outputs=(d,))
        return encoder

    def __get_our_encoder(self, num_nodes, d, n_units, nu1, nu2, activation_function):
        input_layer = layers.Input(shape=(num_nodes,))
        encoded = input_layer
        for unit in n_units:
            encoded = layers.Dense(
                unit,
                activation=activation_function,
                bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            )(encoded)

        encoder_output = layers.Dense(
            d,
            activation=activation_function,
            bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
        )(encoded)

        # encoder = models.Model(inputs=input_layer, outputs=encoder_output)
        return (input_layer, encoder_output)

    def __get_decoder(self, num_nodes, d, n_units, nu1, nu2, activation_function):
        K = len(n_units) + 1
        # Input
        y = layers.Input(shape=(d, 1))
        # Decoder layers
        y_hat = [None] * (K + 1)
        y_hat[K] = y
        for i in range(K - 1, 0, -1):
            y_hat[i] = layers.Dense(
                n_units[i - 1],
                activation=activation_function,
                bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            )(y_hat[i + 1])
        y_hat[0] = layers.Dense(
            num_nodes,
            activation=activation_function,
            bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
        )(y_hat[1])
        # Output
        x_hat = y_hat[0]  # decoder's output is also the actual output
        # Decoder Model
        decoder = models.Model(inputs=y, outputs=x_hat)
        return decoder

    def __get_our_decoder(
        self, num_nodes, d, n_units: list, nu1, nu2, activation_function, encoder_output
    ):
        decoded = encoder_output

        for unit in reversed(n_units):
            decoded = layers.Dense(
                unit,
                activation=activation_function,
                bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            )(decoded)

        decoder_output = layers.Dense(
            num_nodes,
            activation=activation_function,
            bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
        )(decoded)

        return decoder_output

    def __get_autoencoder(self, encoder, decoder, input_shape):
        # Input
        x = layers.Input(shape=input_shape)
        # Generate embedding
        y = encoder(x)
        # Generate reconstruction
        x_hat = decoder(y)
        # Autoencoder Model
        autoencoder = models.Model(inputs=x, outputs=[x_hat, y])
        return autoencoder

    def __get_our_autoencoder(self, input_layer, encoder_output, decoder_output):
        autoencoder = models.Model(
            inputs=input_layer, outputs=[encoder_output, decoder_output]
        )
        return autoencoder

    def __batch_generator_dynae(self, graphs, beta, batch_size, look_back, shuffle):
        T = len(graphs)
        d = graphs[0].shape[0]
        train_size = T - look_back

        if not train_size:
            print("======== RETURNING ==========")
            return
        number_of_batches = (d * train_size) // batch_size

        # if number_of_batches < 0:
        # pdb.set_trace()
        # print("# of batches: %d" % number_of_batches)
        counter = 0
        sample_index = np.arange(d * train_size)
        if shuffle:
            np.random.shuffle(sample_index)

        while True:
            batch_index = sample_index[
                batch_size * counter : batch_size * (counter + 1)
            ]
            X_batch = np.zeros((batch_size, d))
            X_batch2 = np.zeros((batch_size, d))
            for idx, record_id in enumerate(batch_index):
                graph_idx = record_id // d
                node_idx = record_id % d
                X_batch[idx, :] = graphs[graph_idx][node_idx, :].toarray()
                X_batch2[idx] = graphs[graph_idx + 1][node_idx, :].toarray()
            y_batch = np.ones(X_batch2.shape)
            y_batch[X_batch2 != 0] = beta
            y_batch[X_batch2 == 0] = -1
            counter += 1
            # yield [X_batch, X_batch2], y_batch
            yield (X_batch, X_batch2), y_batch
            if counter == number_of_batches:
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0

    def __model_batch_predictor_dynae(self, model, graphs, num_nodes, batch_size):
        look_back = len(graphs)
        d = num_nodes
        # graphs_sp = []
        # for graph in graphs:
        #     graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
        counter = 0
        pred = None
        pred2 = None
        while counter < d // batch_size:
            indices = range(batch_size * counter, batch_size * (counter + 1))
            X = np.zeros((batch_size, d))
            X2 = np.zeros((batch_size, d))
            for idx, record_id in enumerate(indices):
                node_idx = record_id
                try:
                    X[idx, :] = graphs[0][node_idx, :].toarray()
                except Exception as e:
                    print(e.message)
                    # pdb.set_trace()

            next_pred, curr_pred = model.predict(X)
            if counter:
                pred = np.vstack((pred, curr_pred))
                pred2 = np.vstack((pred2, next_pred))
            else:
                pred = curr_pred
                pred2 = next_pred
            counter += 1
        if d % batch_size != 0:

            indices = range(batch_size * counter, d)
            X = np.zeros((len(indices), d))
            X2 = np.zeros((len(indices), d))
            for idx, record_id in enumerate(indices):
                node_idx = record_id
                try:
                    X[idx, :] = graphs[0][node_idx, :].toarray()
                except Exception as e:
                    print(e.message)
                    # pdb.set_trace()
            next_pred, curr_pred = model.predict(X)
            if counter:
                pred = np.vstack((pred, curr_pred))
                pred2 = np.vstack((pred2, next_pred))
            else:
                pred = curr_pred
                pred2 = next_pred
        try:
            return pred, pred2
        except:
            # pdb.set_trace()
            pass


class OurDynAE(layers.Model, EmbedderModel):
    def __init__(self, num_nodes, d, n_units: list, nu1, nu2, activation_function):
        super(OurDynAE, self).__init__()
        self.latent_dim = d
        self.shape = num_nodes

        encoder_layers = []
        encoder_layers.append(
            layers.Dense(
                self.shape,
                activation=activation_function,
                bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            )
        )
        for unit in n_units:
            encoder_layers.append(
                layers.Dense(
                    unit,
                    activation=activation_function,
                    bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                    kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                    activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                )
            )
        encoder_layers.append(
            layers.Dense(
                self.latent_dim,
                activation=activation_function,
                bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            )
        )
        self.encoder = Sequential(encoder_layers)

        decoder_layers = []
        for unit in reversed(n_units):
            decoder_layers.append(
                layers.Dense(
                    unit,
                    activation=activation_function,
                    bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                    kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                    activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                )
            )
        decoder_layers.append(
            layers.Dense(
                self.shape,
                activation=activation_function,
                bias_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                kernel_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
                activity_regularizer=regularizers.L1L2(l1=nu1, l2=nu2),
            )
        )
        self.decoder = Sequential(decoder_layers)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_embedding(self, x):
        return self.encoder(x)


class MLP(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, bias=True, **kwargs) -> None:
        super(MLP, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layer_list = nn.ModuleList()
        self.layer_num = len(n_units) + 1

        self.layer_list.append(nn.Linear(input_dim, n_units[0], bias))
        for i in range(1, self.layer_num):
            self.layer_list.append(nn.Linear(n_units[i - 1], n_units[i], bias))
        self.layer_list.append(nn.Linear(n_units[-1], output_dim, bias))

    def forward(self, x):
        for layer in self.layer_list:
            x = F.relu(layer(x))
        return x


class OurPyTDynAE(nn.Module):
    shape: int
    latent_dim: int
    look_back: int
    encoder: nn.Module
    decoder: nn.Module

    def __init__(
        self,
        shape: int,
        latent_dim: int,
        look_back: int,
        n_units: list,
        bias=True,
        **kwargs,
    ) -> None:
        super(OurPyTDynAE, self).__init__(**kwargs)
        self.shape = shape
        self.latent_dim = latent_dim
        self.look_back = look_back

        self.encoder = MLP(shape * look_back, latent_dim, n_units, bias)
        self.decoder = MLP(latent_dim, shape, reversed(n_units), bias)

    def forward(self, x):
        hx = self.encoder(x)
        x_pred = self.decoder(hx)
        return hx, x_pred
