import json
import logging

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import argparse

from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data_loader import CSVDataset
from plot import make_grid, plot_predictions


class Network(nn.Module):
    def __init__(self, input_size, layers, output):
        super().__init__()
        self.layers = []

        sizes = [input_size]
        sizes.extend([l['size'] for l in layers])

        for i in range(1, len(sizes)):
            self.layers.extend([
                nn.Linear(sizes[i - 1], sizes[i]),
                self.get_activation(layers[i - 1]["function"])
            ])

        self.layers.append(nn.Linear(layers[-1]["size"], output))

        self.runner = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.runner(x)

    def get_activation(self, name):
        if name.lower() == 'tanh':
            return nn.Tanh()
        elif name.lower() == 'sigm':
            return nn.Sigmoid()
        else:
            return nn.ReLU()


class Model:
    def __init__(self, num_epochs, learnig_rate, model, criterion, optimizer):
        self.num_epochs = num_epochs
        self.learnig_rate = learnig_rate
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, dataset, train_dataloader):
        name = dataset.split('.')[0]
        frames = []
        for epoch in range(self.num_epochs):
            for i, (features, labels) in enumerate(train_dataloader):
                y_predicted = self.model(features)
                loss = self.criterion(y_predicted, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # plot
                train_dataset = train_dataloader.dataset
                x_train, x_test, y_train, y_test = get_data_from_datasets(train_dataset, train_dataset)
                xx, yy = make_grid(x_train, x_test, y_train, y_test)
                Z = predict_proba_on_mesh_tensor(self, xx, yy)
                frame = plot_predictions(xx, yy, Z, x_train=x_train, x_test=x_test,
                                         y_train=y_train, y_test=y_test, title=name)
                frames.append(frame)

        frames[0].save(f'{name}.gif', format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=10 * self.num_epochs, loop=0)

    def calc_accuracy(self, test_dataloader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            for features, labels in test_dataloader:
                y_predicted = self.model(features)
                _, predicted = torch.max(y_predicted.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Точность равна {acc} %')

    def predict(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)

                _, predicted = torch.max(output_batch.data, 1)
                logging.debug(predicted)

                all_outputs = torch.cat((all_outputs, predicted), 0)

        return all_outputs

    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)

        self.model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                all_outputs = torch.cat((all_outputs, output_batch), 0)

        return all_outputs

    def predict_proba_tensor(self, test_dataloader):
        self.model.eval()

        with torch.no_grad():
            output = self.model(test_dataloader)

        return output


def get_data_from_datasets(train_dataset, test_dataset):
    x_train = train_dataset.data.astype(np.float32)
    y_train = train_dataset.target.astype(int)

    x_test = test_dataset.data.astype(np.float32)
    y_test = test_dataset.target.astype(int)

    return x_train, x_test, y_train, y_test


def predict_proba_on_mesh_tensor(clf, xx, yy):
    q = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict_proba_tensor(q)[:, 1]
    Z = Z.reshape(xx.shape)
    return Z


def train(params, epochs, dataset_path):
    dataset = pd.read_csv(dataset_path, header=0)
    data = dataset[['f1', 'f2']].values
    target = dataset.target.values
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=42)
    train_dataset = CSVDataset(x_train, y_train)
    test_dataset = CSVDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    learnig_rate = params['learning_rate']
    model = Network(2, params['model'], 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learnig_rate)
    
    model = Model(epochs, learnig_rate, model, criterion, optimizer)
    model.train(dataset_path, train_dataloader)
    model.calc_accuracy(test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('config', type=str,
                        help='configuration JSON file with network description')
    parser.add_argument('epochs', type=int,
                        help='an integer for the epochs count')
    parser.add_argument('dataset', type=str,
                        help='CSV dataset path to read')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    train(cfg, args.epochs, args.dataset)
