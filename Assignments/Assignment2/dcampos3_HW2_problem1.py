import os
import argparse
import sys
import numpy as np
import math
import scipy.sparse as sp

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GCN(nn.Module):
    def __init__(self, features, hidden, classes, dropout, layers=2):
        super(GCN, self).__init__()
        self.gc1 = GCLayer(features, hidden)
        self.gc2 = GCLayer(hidden, classes)
        self.gc3 = None
        self.dropout = dropout
        if layers == 3:
            self.gc2 = GCLayer(hidden, hidden)
            self.gc3 = GCLayer(hidden, classes)  
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        if self.gc3 != None:
            x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class GCLayer(Module):
    def __init__(self, dim_in, dim_out):
        super(GCLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.bias = Parameter(torch.FloatTensor(self.dim_out))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    return np.array(list(map(classes_dict.get, labels)),dtype=np.int32)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(path):
    idx_features_labels = np.genfromtxt("{}/cora.content".format(path),dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/cora.cites".format(path),dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    
def main(args):
    set_seed(args.seed)
    print("Loading Data")
    parser = argparse.ArgumentParser()
    adj, features, labels = load_data(args.data_path)
    print("There are {} examples loaded. Splitting into training, validation, and evaluation".format(len(features)))
    dataset_size = len(labels)
    end_of_training_idx = int(args.train_split * dataset_size)
    end_of_validation_idx = int(((1-args.train_split)/2) * dataset_size)
    training_labels = labels[:end_of_training_idx]
    training_input = features[:end_of_training_idx]
    validation_labels = labels[end_of_training_idx:end_of_training_idx+ end_of_validation_idx]
    validation_input = features[end_of_training_idx:end_of_training_idx+ end_of_validation_idx]
    evaluation_labels = labels[end_of_training_idx+ end_of_validation_idx:]
    evaluation_input = features[end_of_training_idx+ end_of_validation_idx:]
    print("Data Loaded")

    print("Loading Model")
    model = GCN(features=features.shape[1],hidden=args.hidden, classes=labels.max().item() + 1, dropout=args.dropout, layers=args.layers)
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == 'rms':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=args.momentum, centered=False)
    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")
        adj = adj.to("cuda")
        features = features.to("cuda")
        validation_labels = validation_labels.to("cuda")
        validation_input = validation_input.to("cuda")
        training_labels = training_labels.to("cuda")
        training_input = training_input.to("cuda")
        evaluation_labels = evaluation_labels.to("cuda")
        evaluation_input = evaluation_input.to("cuda")
    print("Model Loaded")

    print("Training Model")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)[:end_of_training_idx]
        loss = F.nll_loss(output, training_labels)
        loss.backward()
        optimizer.step()
        #print('Epoch: {:04d}'.format(epoch+1),'loss: {:.4f}'.format(loss.item()))
        model.eval()
        output = model(features, adj)[end_of_training_idx:end_of_training_idx+ end_of_validation_idx]
        loss= F.nll_loss(output, validation_labels)
        acc = accuracy(output, validation_labels)
        #print("Test set results:","loss= {:.4f}".format(loss.item()),"accuracy= {:.4f},".format(acc.item()))
    print("Model Done Training")
    
    print("Evaluation model performance")
    model.eval()
    output = model(features, adj)[end_of_training_idx+ end_of_validation_idx:]
    loss= F.nll_loss(output, evaluation_labels)
    acc = accuracy(output, evaluation_labels)
    #print("Eval set results:","loss= {:.4f}".format(loss.item()),"accuracy= {:.4f},".format(acc.item()))
    print(" & {} & {} \\ \hline".format(loss.item(), acc))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN on CORA')
    parser.add_argument('--optim', default='adam', type=str, help='Optimizer. Options are adam, sgd and rmsprop')
    parser.add_argument('--data_path', default='CS512_HW2_dataset/cora/', type=str, help='Location of CORA File')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA during training.')
    parser.add_argument('--train_split', default=0.6, type=float, help='Percent of samples going to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Length of Training regime')
    parser.add_argument('--lr', type=float, default=1e-2,help='Learning Rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD and RMSprop')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--layers', type=int, default=2, help='layers for the GCN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    args = parser.parse_args()
    main(args)    