import numpy as np
import torch
import torch.nn as nn

from classifier import Classifier


class Net(nn.Module):
    def __init__(self, attributes_cnt, labels_cnt, layers=[200, 100, 50], p=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(attributes_cnt)
        input_size = attributes_cnt
        all_layers = []
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(input_size, labels_cnt))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.layers(x)
        return x


class NeuralNetworkClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self._net = None
        self._epochs = 100
        self._map_label_lblid = {}
        self._map_lblid_label = {}

    def train(self, train_data, train_labels):
        unique_labels = sorted(set(train_labels))
        self._map_label_lblid = {label: idx for idx, label in enumerate(unique_labels)}
        self._map_lblid_label = {idx: label for idx, label in enumerate(unique_labels)}
        train_ids = [self._map_label_lblid[label] for label in train_labels]

        t_train_data = torch.FloatTensor(train_data)
        t_train_labels = torch.tensor(train_ids, dtype=torch.int64)

        labels_cnt = len(set(train_ids))
        atts_cnt = t_train_data.shape[1]

        self._net = Net(atts_cnt, labels_cnt)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=0.001)

        for i in range(self._epochs):
            train_pred = self._net(t_train_data)
            single_loss = loss_function(train_pred, t_train_labels)

            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()

            # print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

    def predict(self, test_data):
        t_test_data = torch.FloatTensor(test_data)

        with torch.no_grad():
            predicted_labels = np.argmax(self._net(t_test_data), axis=1).tolist()

        return [self._map_lblid_label[lblid] for lblid in predicted_labels]
