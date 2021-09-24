import torch
import math
import numpy as np
import os
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
import argparse

from contrib import varprop


def get_dataloader(X, y, params, shuffle):
    data = Data(X,y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=shuffle)
    return dataloader

 
class Data(Dataset):
    def __init__(self, X ,y):
        super(Data, self).__init__()
        self.inputs = X
        self.labels = y

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor([self.inputs[idx]]), torch.tensor([self.labels[idx]])


class UCIDataset(object):

    def __init__(self, name):
        super(UCIDataset, self).__init__()
        file_path = os.path.join('data', name + '.csv')
        data = np.genfromtxt(fname=file_path, delimiter=',',skip_header=1)
        self.X = data[:,:-1]
        self.y = data[:,-1]


class LinearMFS(nn.Module):
    def __init__(self, in_features, out_features, prior_precision=1e0, bias=True, eps=1e-10):
        super(LinearMFS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_precision = prior_precision
        self.eps = eps
        self.has_bias = bias

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_variance = Parameter(torch.ones(in_features) * -10)
        if self.has_bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias_variance = Parameter(torch.ones(out_features) * -10)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs_mean, inputs_variance):
        weight_variance = self.weight_variance.unsqueeze(dim=0).repeat(self.out_features,1)
        outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = F.linear(inputs_mean**2 * torch.exp(self.weight_variance), torch.ones_like(self.weight).to(self.weight_variance.device))
        if inputs_variance is not None:
            outputs_variance += F.linear(inputs_variance, self.weight**2 + torch.exp(weight_variance))
        if self.has_bias:
            outputs_variance += torch.exp(self.bias_variance)
        return outputs_mean, outputs_variance

    def kl_div(self):
        weight_variance = self.weight_variance.unsqueeze(dim=0).repeat(self.out_features,1)
        kld = 0.5 * (-weight_variance + self.prior_precision * (torch.exp(weight_variance) + self.weight**2)).sum()
        if self.has_bias:
            kld += 0.5 * (-self.bias_variance + self.prior_precision * (torch.exp(self.bias_variance) + self.bias**2)).sum()
        return kld



def finitialize(modules):
    for layer in modules:
        if isinstance(layer, (varprop.LinearMN, LinearMFS)):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

def keep_variance(x, min_variance):
    return x.clamp(min=min_variance)


class UCINetMNVI(nn.Module):

    def __init__(self, num_features, hidden_units, prior_precision):
        super(UCINetMNVI, self).__init__()
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=1e-5)
        self.fc1 = varprop.LinearMN(num_features, hidden_units, prior_precision=prior_precision)
        self.fc2 = varprop.LinearMN(hidden_units, 1, prior_precision=prior_precision)
        self.fc3 = varprop.LinearMN(hidden_units, 1, prior_precision=prior_precision)
        self.relu = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)

        finitialize(self.modules())

    def forward(self, inputs):
        inputs_mean = inputs.float()
        inputs_variance = torch.zeros_like(inputs_mean)
        x = self.fc1(inputs_mean, inputs_variance)
        x = self.relu(*x)
        prediction_mean, prediction_variance = self.fc2(*x)
        ale_uncertainty_mean, ale_uncertainty_variance = self.fc3(*x)

        return torch.cat((prediction_mean, prediction_variance,
            ale_uncertainty_mean, ale_uncertainty_variance), 1)
        
    def kl_div(self):
        kl = 0
        for module in self.children():
            if isinstance(module, varprop.LinearMN):
                kl += module.kl_div()
        return kl


class UCINetMFSVI(nn.Module):

    def __init__(self, num_features, hidden_units, prior_precision):
        super(UCINetMFSVI, self).__init__()
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=1e-5)
        self.fc1 = LinearMFS(num_features, hidden_units, prior_precision=prior_precision)
        self.fc2 = LinearMFS(hidden_units, 1, prior_precision=prior_precision)
        self.fc3 = LinearMFS(hidden_units, 1, prior_precision=prior_precision)
        self.relu = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)

        finitialize(self.modules())

    def forward(self, inputs):
        inputs_mean = inputs.float()
        inputs_variance = torch.zeros_like(inputs_mean)
        x = self.fc1(inputs_mean, inputs_variance)
        x = self.relu(*x)
        prediction_mean, prediction_variance = self.fc2(*x)
        ale_uncertainty_mean, ale_uncertainty_variance = self.fc3(*x)

        return torch.cat((prediction_mean, prediction_variance,
            ale_uncertainty_mean, ale_uncertainty_variance), 1)
        
    def kl_div(self):
        kl = 0
        for module in self.children():
            if isinstance(module, LinearMFS):
                kl += module.kl_div()
        return kl




class RegressionLossVI(nn.Module):

    def __init__(self):
        super(RegressionLossVI, self).__init__()

    def forward(self, output, target, params, model):
        target = torch.flatten(target)
        prediction_mean = output[:,0]
        prediction_variance = output[:,1]
        prediction_mean = torch.flatten(prediction_mean)
        prediction_variance = torch.flatten(prediction_variance)
        ale_uncertainty_mean = output[:,2]
        ale_uncertainty_variance = output[:,3]
        ale_uncertainty_mean = torch.flatten(ale_uncertainty_mean)
        ale_uncertainty_variance = torch.flatten(ale_uncertainty_variance)
        diff_square = (prediction_mean - target)**2
        losses = {}
        losses['rmse'] = torch.sqrt(diff_square.mean())
        losses['nelbo'] =  0.5 * (((diff_square + prediction_variance) \
            / torch.exp(ale_uncertainty_mean - 0.5 * ale_uncertainty_variance) \
            + ale_uncertainty_mean).mean() + math.log(2 * math.pi)) \
            + params['kl_div_weight'] * model['net'].kl_div()
        losses['nllh'] = 0.5 * (diff_square / (prediction_variance \
            + torch.exp(ale_uncertainty_mean + 0.5 * ale_uncertainty_variance)) \
            + torch.log(prediction_variance + torch.exp(ale_uncertainty_mean \
            + 0.5 * ale_uncertainty_variance)) + math.log(2 * math.pi)).mean()
        return losses




def train_one_step(inputs, labels, model):
    net = model['net']
    optimizer = model['optimizer']
    criterion = model['criterion']
    net.train()
    optimizer.zero_grad()
    output = net(inputs)
    losses = criterion(output, labels, params, model)
    losses['nelbo'].backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type='inf')
    optimizer.step()
    return losses

def evaluate(model, epoch):
    net = model['net']
    criterion = model['criterion']
    net.eval()
    train_rmse = 0
    train_nllh = 0
    test_rmse = 0
    test_nllh = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(model['training_loader']):
            inputs, labels = inputs.to(model['device']), labels.to(model['device'])
            output = net(inputs)
            losses = criterion(output, labels, params, model)
            train_rmse += losses['rmse']
            train_nllh += losses['nllh']
        train_rmse = (train_rmse / (i+1)).item()
        train_nllh = (train_nllh / (i+1)).item()

        for i, (inputs, labels) in enumerate(model['test_loader']):
            inputs, labels = inputs.to(model['device']), labels.to(model['device'])
            output = net(inputs)
            losses = criterion(output, labels, params, model)
            test_rmse += losses['rmse']
            test_nllh += losses['nllh']
        test_rmse = (test_rmse / (i+1)).item()
        test_nllh = (test_nllh / (i+1)).item()
    print('epoch %3d train_rmse %5.2f test_rmse %5.2f train_nllh %5.2f test_nllh %5.2f' \
        %(epoch, train_rmse, test_rmse, train_nllh, test_nllh))
    return train_rmse, test_rmse, train_nllh, test_nllh


def main():

    train_rmse_runs = []
    test_rmse_runs = []
    train_nllh_runs = []
    test_nllh_runs = []

    model = {}
    model['criterion'] = RegressionLossVI()

    dataset = UCIDataset(params['datasetname'])

    model['device'] = torch.device(params['device'])

    X, y = dataset.X, dataset.y
    params['num_examples'] = len(y)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    y = y.reshape(-1,1)

    for run in range(params['runs']):

        j = 0
        avg_train_rmse = 0
        avg_test_rmse = 0
        avg_train_nllh = 0
        avg_test_nllh = 0
        train_rmse_split = []
        test_rmse_split = []
        train_nllh_split = []
        test_nllh_split = []
        
        kf = KFold(n_splits=params['splits'], shuffle=True, random_state=params['seed'])
        kf.get_n_splits(X)

        for train_index, test_index in kf.split(X):
            if params['net'] == 'MNVI':
                model['net'] = UCINetMNVI(X.shape[1],  hidden_units=params['hidden_units'],
                prior_precision=params['prior_precision'])
            elif params['net'] == 'MFSVI':
                model['net'] = UCINetMFSVI(X.shape[1],  hidden_units=params['hidden_units'],
                prior_precision=params['prior_precision'])
            model['net'].to(model['device'])
            model['optimizer'] = torch.optim.SGD(model['net'].parameters(),
                lr=params['lr'], momentum=0.9)
            model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(model['optimizer'],
                milestones=params['scheduler_milestones'], gamma=params['scheduler_decay'])

            params['kl_div_weight'] =  1 / (100 * (1 - 1 / params['splits']) * params['num_examples'])

            j +=1
            print('Evaluation on split %2d' %j)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            model['training_loader'] = get_dataloader(X_train, y_train, params, shuffle=True)
            model['test_loader'] = get_dataloader(X_test, y_test, params, shuffle=False)
            for epoch in range(params['max_epochs']):
                loss = 0
                for i, (inputs, labels) in enumerate(model['training_loader']):
                    inputs, labels = inputs.to(model['device']), labels.to(model['device'])
                    train_one_step(inputs, labels, model)
                model['scheduler'].step()
                if (epoch + 1) % 10 == 0:
                    train_rmse, test_rmse, train_nllh, test_nllh = evaluate(model, epoch+1)
                if epoch + 1 in params['scheduler_milestones']:
                    params['kl_div_weight'] *= 10
            train_rmse_split.append(train_rmse)
            test_rmse_split.append(test_rmse)
            train_nllh_split.append(train_nllh)
            test_nllh_split.append(test_nllh)
            print()

        for i in range(params['splits']):
            avg_train_rmse += train_rmse_split[i] / params['splits']
            avg_test_rmse += test_rmse_split[i] / params['splits']
            avg_train_nllh += train_nllh_split[i] / params['splits']
            avg_test_nllh += test_nllh_split[i] / params['splits']

        print('avg_train_rmse %5.2f avg_test_rmse %5.2f avg_train_nllh %5.2f avg_test_nllh %5.2f' \
            %(avg_train_rmse, avg_test_rmse, avg_train_nllh, avg_test_nllh))
        print()

        train_rmse_runs.append(avg_train_rmse)
        test_rmse_runs.append(avg_test_rmse)
        train_nllh_runs.append(avg_train_nllh)
        test_nllh_runs.append(avg_test_nllh)


    avg_train_rmse = 0
    avg_test_rmse = 0
    avg_train_nllh = 0
    avg_test_nllh = 0

    std_train_rmse = 0
    std_test_rmse = 0
    std_train_nllh = 0
    std_test_nllh = 0

    for i in range(params['runs']):
            avg_train_rmse += train_rmse_runs[i] / params['runs']
            avg_test_rmse += test_rmse_runs[i] / params['runs']
            avg_train_nllh += train_nllh_runs[i] / params['runs']
            avg_test_nllh += test_nllh_runs[i] / params['runs']

    print('avg_train_rmse %5.2f avg_test_rmse %5.2f avg_train_nllh %5.2f avg_test_nllh %5.2f' \
            %(avg_train_rmse, avg_test_rmse, avg_train_nllh, avg_test_nllh))

    for i in range(params['runs']):
            std_train_rmse += (train_rmse_runs[i] - avg_train_rmse)**2 / (params['runs'] -1)
            std_test_rmse += (test_rmse_runs[i] - avg_test_rmse)**2 / (params['runs'] -1)
            std_train_nllh += (train_nllh_runs[i] - avg_train_nllh)**2 / (params['runs'] -1)
            std_test_nllh += (test_nllh_runs[i] - avg_test_nllh)**2 / (params['runs'] -1)

    print('std_train_rmse %5.2f std_test_rmse %5.2f std_train_nllh %5.2f std_test_nllh %5.2f' \
            %(math.sqrt(std_train_rmse), math.sqrt(std_test_rmse), math.sqrt(std_train_nllh), math.sqrt(std_test_nllh)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetname', default='')
    parser.add_argument('--net', default='MNVI')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--scheduler_milestones', nargs='+', type=int, default=[100, 150])
    parser.add_argument('--scheduler_decay', default=0.1, type=float)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--splits', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--prior_precision', default=1e0, type=float)
    params = vars(parser.parse_args())
    main()