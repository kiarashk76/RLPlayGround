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


counterrrr = [0, 0, 0, 0]
#experiment configs
num_runs = 1
num_epochs = 50
num_data_points = 100
plt_show = True
plot_show_epoch_freq = 100
fetch_batch_size = 16

#agents configs
num_agents = 1
names = ['2']
num_networks = [2]
hidden_layers_sizes = [[]]
batch_sizes = [16]
step_sizes = [0.01]
choose_step_sizes = [0.01]
plot_colors = ['g']

assert num_agents == len(names) == len(num_networks) == len(hidden_layers_sizes) \
       == len(batch_sizes) == len(step_sizes) == len(plot_colors) == len(choose_step_sizes)\
    , "inconsistency in agent configurations"

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
class ModelChoosing(nn.Module):
    def __init__(self, num_models):
        super(ModelChoosing, self).__init__()
        self.l1 = nn.Linear(1, 32)
        self.head = nn.Linear(32, num_models)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        return torch.softmax(self.head(x), dim=1)

class GeneralModel():
    def __init__(self, params):
        self.num_networks = params['num_networks']
        self.hidden_layers = params['hidden_layers']
        self.batch_size = params['batch_size']
        self.step_size = params['step_size']
        self.choose_step_size = params['choose_step_size']
        self.name = params['name']
        self.models = []

        self.batch_x = np.zeros([self.num_networks, self.batch_size])
        self.batch_y = np.zeros([self.num_networks, self.batch_size])
        self.batch_counter = np.zeros([self.num_networks], dtype=int)

        self.choosing_batch_size = 16
        self.choosing_batch_x = np.zeros([self.choosing_batch_size])
        self.choosing_batch_y = np.zeros([self.choosing_batch_size, self.num_networks])
        self.choosing_batch_counter = 0
        self.__create_networks()

    def __create_networks(self):
        for i in range(self.num_networks):
            self.models.append(StateTransitionModel(self.hidden_layers))
        self.choosing_network = ModelChoosing(self.num_networks)

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
        # print(m)
        for c, index in enumerate(m):
            self.batch_x[index, self.batch_counter[index]] = X[c]
            self.batch_y[index, self.batch_counter[index]] = y[c]
            if index == 0 and X[c] < 0:
                counterrrr[0] += 1
            elif index == 0 and X[c] > 0:
                counterrrr[1] += 1
            elif index == 1 and X[c] < 0:
                counterrrr[2] += 1
            elif index == 1 and X[c] > 0:
                counterrrr[3] += 1
            else:
                raise ValueError("khar")
            self.__train_choosing_network(X[c], index)
            self.batch_counter[index] += 1
            if self.batch_counter[index] == self.batch_size:
                self.__train_model(index)
                self.batch_counter[index] = 0

    def __train_choosing_network(self, x, ind):
        y = self.one_hot_encode(ind, self.num_networks)
        self.choosing_batch_x[self.choosing_batch_counter] = x
        self.choosing_batch_y[self.choosing_batch_counter] = y
        self.choosing_batch_counter += 1
        if self.choosing_batch_counter == self.choosing_batch_size:
            x = torch.from_numpy(self.choosing_batch_x.reshape(self.choosing_batch_size,1)).float()
            y = torch.from_numpy(self.choosing_batch_y).float()
            y_hat = self.choosing_network(x)
            assert y_hat.shape == y.shape
            loss = torch.mean((y_hat - y)**2)
            loss.backward()

            optimizer = optim.Adam(self.choosing_network.parameters(), lr=self.choose_step_size)
            # optimizer = optim.SGD(model.parameters(), lr=step_size)

            optimizer.step()
            optimizer.zero_grad()
            self.choosing_batch_counter = 0

    def choose_network(self, x):
        with torch.no_grad():
            y_hat = self.choosing_network(x)
            m = np.argmax(y_hat, axis= 1)
        return m

    def one_hot_encode(self, index, total):
        res = np.zeros([total])
        res[index] = 1
        return res

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
        dist_list_true = np.zeros([self.num_networks, len(batch_x)])
        mu_each = []
        sigma_each = []


        for ind, model in enumerate(self.models):
            # choose which model should be used
            mu, sigma = self.__model_output(X, ind)
            dist = torch.abs(y - mu)
            dist_list_true[ind] = dist[:, 0]
            dist_list[ind] = sigma[:, 0]
            mu_each.append(mu)
            sigma_each.append(sigma)
        m_true = np.argmin(dist_list_true, axis=0)
        m_var = np.argmin(dist_list, axis=0)
        m = self.choose_network(X)
        # print(np.mean((m.numpy() - m_true) ** 2))
        for c, index in enumerate(m):
            mu, sigma = mu_each[index][c], sigma_each[index][c]
            mu_list[c] = mu
            sigma_list[c] = sigma

        return mu_list, sigma_list

def drawPlotUncertainty(x, y, y_err, label, color, title=""):
    plt.plot(x, y, color, label=label)
    plt.title(title)
    plt.fill_between(x,
                     y - y_err,
                     y + y_err,
                     facecolor=color, alpha=0.2, edgecolor='none')

if __name__ == "__main__":
    error_list = np.zeros([num_runs, num_epochs, num_agents])
    for r in range(num_runs):
        np.random.seed(r)

        #initializing the models
        models = []
        for i in range(num_agents):
            params = {"num_networks": num_networks[i],
                      "hidden_layers": hidden_layers_sizes[i],
                      "batch_size": batch_sizes[i],
                      "step_size": step_sizes[i],
                      "name": names[i],
                      "choose_step_size": choose_step_sizes[i]}
            m = GeneralModel(params)
            models.append(m)



        # create the dataset
        range_data_points = (-2, 2)
        x = np.random.uniform(range_data_points[0], range_data_points[1], num_data_points)
        x = np.reshape(np.sort(x), (num_data_points, 1))
        # y = x + np.sin(4 * x) + np.sin(13 * x)
        y = np.abs(x)

        #train
        for e in tqdm(range(num_epochs)):
            # train models
            for i in range(num_data_points // fetch_batch_size):
                ind = np.random.choice(num_data_points, fetch_batch_size)
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
                    # mu, var = model.test_model(x, y)
                    drawPlotUncertainty(x[:, 0], mu[:, 0], var[:, 0], 'model'+ model.name, plot_colors[a],
                                        title="epoch "+str(e))
                    plt.plot(x, y, 'black', label='ground truth')

            if e % plot_show_epoch_freq == 0 and plt_show:
                plt.legend()
                plt.show()


        #test
        for a, model in enumerate(models):
            mu, var = model.test_model(x, y)
            # draw plot
            if plt_show:
                drawPlotUncertainty(x[:, 0], mu[:, 0], var[:, 0], 'model' + model.name, plot_colors[a], title='full')
                plt.plot(x, y, 'black', label='ground truth')
        if plt_show:
            plt.legend()
            plt.show()

    # with open('Colab/Logs/log.npy', 'wb') as f:
    #     np.save(f, error_list)

    for i in range(num_agents):
        err = np.mean(error_list, axis=0)[:, i]
        err_bar = np.std(error_list, axis=0)[:, i]
        drawPlotUncertainty(range(len(err)), err, err_bar, models[i].name, plot_colors[i])
    plt.legend()
    plt.show()




print(counterrrr)

