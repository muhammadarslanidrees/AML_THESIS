import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython import display

def Pre_Processing(input_file_path, input_file_name, output_file_path, Class_label, Splits):
    input_data = pd.read_csv(input_file_path+'\\'+ input_file_name)
    
    if('Unnamed: 0' in input_data.columns):
        input_data = input_data.drop(columns = ['Unnamed: 0'])
    
    input_data.fillna(0, inplace=True)

    print('Input Shape: ' + str(input_data.shape))
    print('Input Classes: ' + str(input_data[Class_label].value_counts()))
    
    features = input_data.shape[1]-1
    
    fraud_index = input_data[input_data[Class_label] == 1].index
    non_fraud_index = input_data[input_data[Class_label] == 0].index
    
    non_fraud_start = 0
    non_fraud_end = round(input_data[Class_label].value_counts()[0]/Splits).astype(int)
    non_fraud_increment = round(input_data[Class_label].value_counts()[0]/Splits).astype(int)

    fraud_start = 0
    fraud_end = round(input_data[Class_label].value_counts()[1]/Splits).astype(int)
    fraud_increment = round(input_data[Class_label].value_counts()[1]/Splits).astype(int)
    
    for i in range(0,Splits):
        fraud = input_data.iloc[fraud_index[fraud_start:fraud_end]]
        non_fraud = input_data.iloc[non_fraud_index[non_fraud_start:non_fraud_end]]
        data = fraud.append(non_fraud)
        
        X = data.drop(columns = [Class_label])
        y = data[Class_label]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.2, random_state=42)

        X_train.to_csv(output_file_path + '\\' + str(i) + '_' + 'X_train_' + str(non_fraud_start) + '_' + str(non_fraud_end) + '.csv')
        X_test.to_csv(output_file_path + '\\' + str(i) + '_' + 'X_test_' + str(non_fraud_start) + '_' + str(non_fraud_end) + '.csv')
        y_train.to_csv(output_file_path + '\\' + str(i) + '_' + 'y_train_' + str(non_fraud_start) + '_' + str(non_fraud_end) + '.csv')
        y_test.to_csv(output_file_path + '\\' + str(i) + '_' + 'y_test_' + str(non_fraud_start) + '_' + str(non_fraud_end) + '.csv')
        
        non_fraud_start = non_fraud_end
        fraud_start = fraud_end
        non_fraud_end = non_fraud_start + non_fraud_increment
        fraud_end = fraud_start + fraud_increment

    return features

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu', use_bn=False):
        super(LinearLayer, self).__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU() if act == 'relu' else act
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        if self.use_bn:
            return self.bn(self.act(self.lin(x)))
        return self.act(self.lin(x))

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)

class BaseModel(nn.Module):
    
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(BaseModel, self).__init__()
        self.f1 = Flatten()
        self.lin1 = LinearLayer(num_inputs, num_hidden, use_bn=True)
        self.lin2 = LinearLayer(num_hidden, num_hidden, use_bn=True)
        self.lin3 = nn.Linear(num_hidden, num_outputs)
        
    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(self.f1(x))))

class CramerRaoLowerBound:

    def __init__(self, model, crit, penalty, weight_decay_param, lr=0.001, weight=1000000):
        self.model = model
        self.weight = weight
        self.crit = crit
        if penalty in ('SGD', 'CRLBO'):
            self.optimizer = optim.SGD(self.model.parameters(), lr)
        elif penalty == 'L2':
            self.optimizer = optim.SGD(self.model.parameters(), lr, weight_decay=weight_decay_param)

    def forward_backward_update(self, input, target, penalty):
        output = self.model(input)
        if penalty in ('SGD', 'L2'):
            loss = self.crit(output, target)
        elif penalty == 'CRLBO':
            loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
        
    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (input, target) in enumerate(dl):
            if i > num_batch:
                break
            output = F.log_softmax(self.model(input.float()), dim=1)
            log_liklihoods.append(output[:, target.long()])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_crlbo_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params(dataset, batch_size, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0


def accu(model, dataloader):
    model = model.eval()
    acc = 0
    for input, target in dataloader:
        o = model(input.float())
        acc += (o.argmax(dim=1).long() == target.long()).float().mean()
    return acc / len(dataloader)

def read_batch(batch_num, output_file_path):
        
    X_train = pd.read_csv(glob.glob(output_file_path + '\\' + str(batch_num) + '_' + 'X_train_*.csv')[0])
    X_test = pd.read_csv(glob.glob(output_file_path + '\\' + str(batch_num) + '_' + 'X_test_*.csv')[0])
    y_train = pd.read_csv(glob.glob(output_file_path + '\\' + str(batch_num) + '_' + 'y_train_*.csv')[0], header = None, names=['Index', 'Class'])
    y_test = pd.read_csv(glob.glob(output_file_path + '\\' + str(batch_num) + '_' + 'y_test_*.csv')[0], header = None, names=['Index', 'Class'])
    
    if('Unnamed: 0' in X_train.columns):
        X_train = X_train.drop(columns = ['Unnamed: 0'])
        
    if('Unnamed: 0' in X_test.columns):
        X_test = X_test.drop(columns = ['Unnamed: 0'])    
    
    X_train = X_train.drop(X_train.index[0:X_train.shape[0]%10])
    y_train = y_train.drop(y_train.index[0:y_train.shape[0]%10])
    
    X_test = X_test.drop(X_test.index[0:X_test.shape[0]%10])
    y_test = y_test.drop(y_test.index[0:y_test.shape[0]%10])
        
    mini_batches = round(X_train.shape[0]/100)
    mini_batches = mini_batches - (mini_batches%10)
    
    X = torch.from_numpy(np.array(X_train.astype('float')))
    y = torch.tensor(np.array(y_train['Class']), dtype=torch.long)

    train_set = torch.utils.data.TensorDataset(X, y)

    X_t = torch.from_numpy(np.array(X_test.astype('float')))
    y_t = torch.tensor(np.array(y_test['Class']), dtype=torch.long)

    test_set = torch.utils.data.TensorDataset(X_t, y_t)

    train_loader = DataLoader(train_set, batch_size = 100, shuffle=True)
    test_loader = DataLoader(test_set, batch_size = 100, shuffle=False)

    return train_loader, test_loader, train_set, mini_batches	
