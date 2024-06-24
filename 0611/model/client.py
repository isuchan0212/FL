import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import copy
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Tuple
from sklearn.model_selection import train_test_split
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model.gcn import GCN


class Client(torch.nn.Module):
    def __init__(self, args, subgraphs):
        super().__init__()
        self.args = args

        base_size = len(subgraphs) // self.args.n_clients
        remainder = len(subgraphs) % self.args.n_clients
        self.subgraphs = []

        for i in range(self.args.n_clients):
            size = base_size + (1 if i < remainder else 0)
            self.subgraphs.append(DataLoader(subgraphs[i*size:(i+1)*size], batch_size=self.args.batch_size, shuffle=True)) # shuffle 

        self.model = GCN(166, [100])


    def update(self, global_state, client_id):
        self.model.load_state_dict(global_state[client_id])

    def train(self, client_id, global_state, global_train_result, cur_round, gpu, lock):
        if global_state[client_id] is not None:
            self.update(global_state, client_id)
        self.model.to(gpu)

        subgraph = self.subgraphs[client_id]

        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7,0.3]).to(gpu))

        for epoch in range(self.args.local_epochs):
            train_loss = 0.0

            for batch in subgraph:
                batch = batch.to(gpu)
                optimizer.zero_grad()

                out = self.model(batch)
                loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                train_loss += loss.item() * batch.num_graphs
                optimizer.step()

            train_loss /= len(self.subgraphs[client_id])

        global_state[client_id] = self.model.state_dict()
        self.transfer_to_server(global_state, client_id, lock)
        global_train_result[client_id][cur_round] = train_loss / self.args.local_epochs

        print(train_loss/ self.args.local_epochs)
        print(global_train_result[client_id])

    def val(self, client_id, global_state, global_val_result, cur_round, gpu):
        self.update(global_state, client_id)
        self.model.to(gpu)

        subgraph = self.subgraphs[client_id]

        self.model.eval()
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7,0.3]).to(gpu))

        with torch.no_grad():
            val_loss = 0.0
            for batch in subgraph:
                batch = batch.to(gpu)

                out = self.model(batch)
                loss = loss_fn(out[batch.val_mask], batch.y[batch.val_mask])
                val_loss += loss.item() * batch.num_graphs

            val_loss /= len(self.subgraphs[client_id])

        global_val_result[client_id][cur_round] = val_loss

        print(val_loss)
        print(global_val_result[client_id])

    def test(self, client_id, global_state, global_test_result, cur_round, gpu):
        self.update(global_state, client_id)
        self.model.to(gpu)

        subgraph = self.subgraphs[client_id]

        self.model.eval()
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7,0.3]).to(gpu))

        with torch.no_grad():
            test_loss = 0.0
            for batch in subgraph:
                batch = batch.to(gpu)

                out = self.model(batch)
                loss = loss_fn(out[batch.test_mask], batch.y[batch.test_mask])
                test_loss += loss.item() * batch.num_graphs

            test_loss /= len(self.subgraphs[client_id])

        global_test_result[client_id][cur_round] = test_loss

        print(test_loss)
        print(global_test_result[client_id])

    def transfer_to_server(self, global_state, client_id, lock):
        with lock :
            global_state[client_id] = self.model.state_dict()
    

    

    