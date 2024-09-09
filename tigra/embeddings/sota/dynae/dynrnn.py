from typing import List
import numpy as np
from tigra.embeddings.sota.dynae.ae_embedding_model import (
    DynamicAutoEncoderEmbeddingModel,
)
from tigra.embeddings.sota.dynae.dnn_utils import WeightedMSEX
import tensorflow as tf
from keras import Sequential, layers, models, optimizers


class DynRNN(DynamicAutoEncoderEmbeddingModel):
    def __init__(self, d, **kwargs) -> None:
        """Initialize the dynamic RNN class

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
        super().__init__(d, method_name="DynRNN", **kwargs)

    def learn_embedding(self, graphs: list):
        num_nodes = graphs[0].shape[0]
        encoder = self.__get_lstm_encoder_v2(
            num_nodes,
            self._n_prev_graphs,
            self._d,
            self._n_enc_units,
            self._actfn,
            self._nu1,
            self._nu2,
            None,
            None,
            None,
        )
        decoder = self.__get_lstm_decoder_v2(
            num_nodes,
            self._n_prev_graphs,
            self._d,
            self._n_enc_units,
            self._actfn,
            self._nu1,
            self._nu2,
            None,
            None,
            None,
        )
        autoencoder = self.__get_lstm_autoencoder(
            encoder, decoder, (self._n_prev_graphs, num_nodes)
        )

        # Initialize self._model
        # Input
        x_in = layers.Input(shape=(self._n_prev_graphs, num_nodes), name="x_in")
        x_pred = layers.Input(shape=(num_nodes,), name="x_pred")
        # Process inputs
        [x_hat, y] = autoencoder(x_in)
        # Outputs
        x_diff = layers.Subtract()([x_hat, x_pred])

        #         x_diff = merge([x_hat, x_pred],
        #                        mode=lambda a, b: a - b,
        #                        output_shape=lambda L: L[1])

        # Objectives
        # def weighted_mse_x(y_true, y_pred):
        #     ''' Hack: This fn doesn't accept additional arguments.
        #               We use y_true to pass them.
        #         y_pred: Contains x_hat - x_pred
        #         y_true: Contains b
        #     '''
        #     return KBack.sum(
        #         KBack.square(y_pred * y_true[:, 0:self._node_num]),
        #         axis=-1
        #     )

        # Model
        model = models.Model(inputs=[x_in, x_pred], outputs=x_diff)
        sgd = optimizers.SGD(
            learning_rate=self._xeta, weight_decay=1e-5, momentum=0.99, nesterov=True
        )
        adam = optimizers.Adam(
            learning_rate=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )
        model.compile(optimizer=adam, loss=WeightedMSEX(num_nodes))
        # self._model.compile(optimizer='rmsprop', loss=weighted_mse_x)
        history = model.fit(
            self.__batch_generator_dynrnn(
                graphs, self._beta, self._n_batch, self._n_prev_graphs, True
            ),
            epochs=self._n_iter,
            steps_per_epoch=(num_nodes * self._n_prev_graphs) // self._n_batch,
            verbose=1,
            # callbacks=[tensorboard]
        )
        loss = history.history["loss"]
        # Get embedding for all points
        if loss[0] == np.inf or np.isnan(loss[0]):
            print("Model diverged. Assigning random embeddings")
            self._Y = np.random.randn(num_nodes, self._d)
        else:
            self._Y, self._next_adj = self.__model_batch_predictor_dynrnn(
                autoencoder,
                graphs[len(graphs) - self._n_prev_graphs :],
                self._n_batch,
            )

    def get_embedding(self) -> List:
        return self._Y

    def __get_lstm_encoder_v2(
        self,
        n_nodes,
        look_back,
        d,
        n_units,
        activation_fn,
        nu1,
        nu2,
        bias_reg,
        input_reg,
        recurr_reg,
        ret_seq=True,
    ):
        model = Sequential()
        model.add(layers.Input(shape=(look_back, n_nodes)))
        # model.add(layers.Dense(d, input_shape=(look_back, n_nodes),
        #             activation=activation_fn,
        #              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
        #              )
        n_rnn_layers = len(n_units)
        model.add(
            layers.LSTM(
                n_units[0],
                # input_shape=(look_back, n_nodes),
                return_sequences=True,
                bias_regularizer=bias_reg,
                kernel_regularizer=input_reg,
                recurrent_regularizer=recurr_reg,
            )
        )
        for l_idx, n_unit in enumerate(n_units[1:]):
            model.add(
                layers.LSTM(
                    n_unit,
                    return_sequences=True,
                    bias_regularizer=bias_reg,
                    kernel_regularizer=input_reg,
                    recurrent_regularizer=recurr_reg,
                )
            )
        model.add(
            layers.LSTM(
                d,
                return_sequences=True,
                bias_regularizer=bias_reg,
                kernel_regularizer=input_reg,
                recurrent_regularizer=recurr_reg,
            )
        )
        return model

    def __get_lstm_decoder_v2(
        self,
        n_nodes,
        look_back,
        d,
        n_units,
        activation_fn,
        nu1,
        nu2,
        bias_reg,
        input_reg,
        recurr_reg,
    ):
        model = Sequential()
        n_rnn_layers = len(n_units)
        model.add(layers.Input(shape=(look_back, d)))
        model.add(
            layers.LSTM(
                d,
                # input_shape=(look_back, d),
                # input_shape=(1, n_nodes),
                return_sequences=True,
                bias_regularizer=bias_reg,
                kernel_regularizer=input_reg,
                recurrent_regularizer=recurr_reg,
            )
        )
        for l_idx, n_unit in enumerate(n_units[::-1]):
            if l_idx < n_rnn_layers - 1:
                model.add(
                    layers.LSTM(
                        n_unit,
                        return_sequences=True,
                        bias_regularizer=bias_reg,
                        kernel_regularizer=input_reg,
                        recurrent_regularizer=recurr_reg,
                    )
                )
            else:
                model.add(
                    layers.LSTM(
                        n_nodes,
                        return_sequences=False,
                        bias_regularizer=bias_reg,
                        kernel_regularizer=input_reg,
                        recurrent_regularizer=recurr_reg,
                    )
                )

        # model.add(Dense(n_nodes, activation=activation_fn,
        #              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
        #              )
        return model

    def __get_lstm_autoencoder(self, encoder, decoder, input_shape):
        # Input

        x = layers.Input(shape=input_shape)
        # Generate embedding
        y = encoder(x)
        # Generate reconstruction
        x_hat = decoder(y)
        # Autoencoder Model
        autoencoder = models.Model(inputs=x, outputs=[x_hat, y])
        return autoencoder

    def __batch_generator_dynrnn(self, graphs, beta, batch_size, look_back, shuffle):
        T = len(graphs)
        n = graphs[0].shape[0]
        train_size = T - look_back
        number_of_batches = (n * train_size) // batch_size
        counter = 0
        sample_index = np.arange(n * train_size)
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[
                batch_size * counter : batch_size * (counter + 1)
            ]
            X_batch = np.zeros((batch_size, look_back, n))
            X_batch2 = np.zeros((batch_size, n))
            for idx, record_id in enumerate(batch_index):
                graph_idx = record_id // n
                node_idx = record_id % n
                # X_batch[idx, :] = graphs_sp[graph_idx][node_idx, :].toarray()
                # X_batch2[idx] = graphs_sp[graph_idx+1][node_idx, :].toarray()
                for tau in range(look_back):
                    X_batch[idx, tau, :] = graphs[graph_idx + tau][
                        node_idx, :
                    ].toarray()
                X_batch2[idx] = graphs[graph_idx + look_back][node_idx, :].toarray()
            y_batch = beta * np.ones(X_batch2.shape)
            y_batch[X_batch2 != 0] = beta
            y_batch[X_batch2 == 0] = -1
            # y_batch[X_batch2 == 0] = 0
            counter += 1
            # np.savetxt('debug_x_batch.txt',
            #                X_batch.reshape((batch_size, d)))
            # np.savetxt('debug_x_batch2.txt',
            #                X_batch.reshape((batch_size, d)))
            # np.savetxt('debug_y_batch.txt',
            #                y_batch)
            yield [X_batch, X_batch2], y_batch
            if counter == number_of_batches:
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0

    def __model_batch_predictor_dynrnn(self, model, graphs, batch_size):
        num_nodes = graphs[0].shape[0]
        n_samples = num_nodes
        look_back = len(graphs)
        d = num_nodes
        counter = 0
        pred = None
        pred2 = None
        while counter < n_samples // batch_size:
            indices = range(batch_size * counter, batch_size * (counter + 1))
            X = np.zeros((batch_size, look_back, d))
            X2 = np.zeros((batch_size, d))
            for idx, record_id in enumerate(indices):
                node_idx = record_id
                for tau in range(look_back):
                    X[idx, tau, :] = graphs[tau][node_idx, :].toarray()

            next_pred, curr_pred = model.predict(X)
            if counter:
                pred = np.vstack((pred, curr_pred))
                pred2 = np.vstack((pred2, next_pred))
            else:
                pred = curr_pred
                pred2 = next_pred
            counter += 1
        if n_samples % batch_size != 0:
            indices = range(batch_size * counter, n_samples)
            X = np.zeros((len(indices), look_back, d))
            X2 = np.zeros((len(indices), d))
            for idx, record_id in enumerate(indices):
                node_idx = record_id
                for tau in range(look_back):
                    X[idx, tau, :] = graphs[tau][node_idx, :].toarray()
            next_pred, curr_pred = model.predict(X)
            if counter:
                pred = np.vstack((pred, curr_pred))
                pred2 = np.vstack((pred2, next_pred))
            else:
                pred = curr_pred
                pred2 = next_pred
        try:
            # return pred, pred2
            return pred[:, -1, :].reshape((pred.shape[0], pred.shape[2])), pred2
        except:
            # pdb.set_trace()
            pass
