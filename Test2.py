import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_runs = 5
num_agents = 2
plt_show = False
num_data_points = 2000
num_epochs = 20
range_data_points = (-1, 2)
plot_show_epoch_freq = 10
batch_size = 16
plot_colors = ['r', 'g']

class StateTransitionModel(nn.Module):
    def __init__(self,num_hidden):
        super(StateTransitionModel, self).__init__()
        '''
        self.layer1 = nn.Linear(1,64)
        self.layer2 = nn.Linear(64,64)
        self.mu = nn.Linear(64,1)

        self.vlayer1 = nn.Linear(1, 64)
        self.vlayer2 = nn.Linear(64, 64)
        self.var = nn.Linear(64, 1)

        torch.nn.init.xavier_uniform_(self.vlayer1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.vlayer2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.layer1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.layer2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)
        '''
        self.layers_list = []
        self.vlayers_list = []
        if len(num_hidden) == 0:
            self.mu = nn.Linear(1, 1)
            self.var = nn.Linear(1, 1)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)
        else:
            for i, num in enumerate(num_hidden):
                if i == 0:
                    l = nn.Linear(1, num)
                    torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                    self.layers_list.append(l)
                    self.add_module('hidden_layer_' + str(i), l)

                    vl = nn.Linear(1, num)
                    torch.nn.init.xavier_uniform_(vl.weight, gain=1.0)
                    self.vlayers_list.append(vl)
                    self.add_module('vhidden_layer_' + str(i), vl)
                else:
                    l = nn.Linear(num_hidden[i-1], num)
                    torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                    self.layers_list.append(l)
                    self.add_module('hidden_layer_' + str(i), l)

                    vl = nn.Linear(num_hidden[i - 1], num)
                    torch.nn.init.xavier_uniform_(vl.weight, gain=1.0)
                    self.vlayers_list.append(vl)
                    self.add_module('vhidden_layer_' + str(i), vl)
            self.mu = nn.Linear(num_hidden[-1], 1)
            self.var = nn.Linear(num_hidden[-1], 1)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)


    def forward(self, x):
        '''
        l = torch.relu(self.layer1(x))
        l = torch.relu(self.layer2(l))
        vl = torch.relu(self.vlayer1(x))
        vl = torch.relu(self.vlayer2(vl))

        '''
        l = x
        vl = x
        for i, lay in enumerate(zip(self.layers_list, self.vlayers_list)):
            layer = lay[0]
            vlayer = lay[1]
            l = torch.relu(layer(l))
            vl = torch.relu(vlayer(vl))
        mu = self.mu(l)
        var = torch.log(1 + torch.exp(self.var(vl)))
        return mu, var

class GeneralModel():
    def __init__(self, params):
        self.num_networks = params['num_networks']
        self.hidden_layers = params['hidden_layers']
        self.batch_size = params['batch_size']
        self.step_size = params['step_size']
        self.name = params['name']
        self.models = []

        self.batch_x = np.zeros([self.num_networks, self.batch_size])
        self.batch_y = np.zeros([self.num_networks, self.batch_size])
        self.batch_counter = np.zeros([self.num_networks], dtype=int)

        self.__create_networks()

    def __create_networks(self):
        for i in range(self.num_networks):
            self.models.append(StateTransitionModel(self.hidden_layers))

    def __model_output(self, batch_x, ind):
        with torch.no_grad():
            mu, sigma = self.models[ind](batch_x.float())
        return mu, sigma

    def train_model(self, batch_x, batch_y):
        X = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()
        dist_list = np.zeros([self.num_networks, len(batch_x)])
        for ind, model in enumerate(self.models):
            mu, sigma = self.__model_output(X, ind)
            dist = torch.abs(y - mu)
            dist_list[ind] = dist[:, 0]
        m = np.argmin(dist_list, axis=0)
        for c, index in enumerate(m):
            self.batch_x[index, self.batch_counter[index]] = X[c]
            self.batch_y[index, self.batch_counter[index]] = y[c]
            self.batch_counter[index] += 1
            if self.batch_counter[index] == self.batch_size:
                self.__train_model(index)
                self.batch_counter[index] = 0

    def __train_model(self, model_ind):
        x = torch.from_numpy(self.batch_x[model_ind].reshape(self.batch_size,1)).float()
        y = torch.from_numpy(self.batch_y[model_ind].reshape(self.batch_size,1)).float()
        pred, var = self.models[model_ind](x)
        assert pred.shape == var.shape == x.shape, str(pred.shape) + str(var.shape) + str(x.shape)
        loss = torch.mean(((pred - y) ** 2) / (2 * var) + 0.5 * torch.log(var))
        loss.backward()

        optimizer = optim.Adam(self.models[model_ind].parameters(), lr=self.step_size)
        # optimizer = optim.SGD(model.parameters(), lr=step_size)

        optimizer.step()
        optimizer.zero_grad()

    def test_model(self, batch_x, batch_y):
        X = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()
        mu_list = np.zeros_like(batch_x)
        sigma_list = np.zeros_like(batch_x)
        dist_list = np.zeros([self.num_networks, len(batch_x)])
        mu_each = []
        sigma_each = []

        for ind, model in enumerate(self.models):
            # choose which model should be used
            mu, sigma = self.__model_output(X, ind)
            dist = torch.abs(y - mu)
            dist_list[ind] = dist[:, 0]
            mu_each.append(mu)
            sigma_each.append(sigma)

        m = np.argmin(dist_list, axis=0)
        for c, index in enumerate(m):
            mu, sigma = mu_each[index][c], sigma_each[index][c]
            mu_list[c] = mu
            sigma_list[c] = sigma

        return mu_list, sigma_list

def drawPlotUncertainty(x, y, y_err, label, color):
    plt.plot(x, y, color, label=label)
    plt.fill_between(x,
                     y - y_err,
                     y + y_err,
                     facecolor=color, alpha=0.2, edgecolor='none')

if __name__ == "__main__":
    error_list = np.zeros([num_runs, num_epochs, num_agents])
    for r in range(num_runs):
        np.random.seed(r)

        #initializing the models
        models = [GeneralModel({"num_networks":1, "hidden_layers":[64, 64, 64],"batch_size":16, "step_size":0.001, "name":'1'}),
                 GeneralModel({"num_networks": 4, "hidden_layers": [32, 32, 32], "batch_size": 16, "step_size": 0.001, "name":"4"})]


        # create the dataset
        x = np.random.uniform(range_data_points[0], range_data_points[1], num_data_points)
        x = np.reshape(np.sort(x), (num_data_points, 1))
        y = x + np.sin(4 * x) + np.sin(13 * x)

        #train
        for e in tqdm(range(num_epochs)):
            # train models
            for i in range(num_data_points // batch_size):
                ind = np.random.choice(num_data_points, batch_size)
                batch_x, batch_y = x[ind], y[ind]
                for model in models:
                    model.train_model(batch_x, batch_y)

            #validate models
            for a, model in enumerate(models):
                mu, var = model.test_model(x, y)
                distance = torch.dist(torch.from_numpy(y).float(), torch.from_numpy(mu).float())
                error_list[r, e, a] = distance

                # draw plot till now
                if e % plot_show_epoch_freq == 0 and plt_show:
                    mu, var = model.test_model(x, y)
                    drawPlotUncertainty(x[:, 0], mu[:, 0], var[:, 0], 'model'+ model.name, plot_colors[a])
                    plt.plot(x, y, 'black', label='ground truth')

            if e % plot_show_epoch_freq == 0 and plt_show:
                plt.legend()
                plt.show()


    with open('log.npy', 'wb') as f:
        np.save(f, error_list)

    for i in range(num_agents):
        err = np.mean(error_list, axis=0)[:, i]
        err_bar = np.std(error_list, axis=0)[:, i]
        drawPlotUncertainty(range(len(err)), err, err_bar, models[i].name, plot_colors[i])
    plt.legend()
    plt.show()






