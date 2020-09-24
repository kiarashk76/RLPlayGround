import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_runs = 1
num_data_points = 5000
num_epochs = 300
range_data_points = (-1, 2)
plot_show_epoch_freq = 1

#single model config
num_hidden_units = 2048
batch_size_single = 16
step_size_single = 0.001

#multi model config
num_hidden_units_multi = 2048
multi_model_num = 4
batch_size_multi = 16
step_size_multi = 0.001

class StateTransitionModel(nn.Module):
    def __init__(self,num_hidden):
        super(StateTransitionModel, self).__init__()
        self.layer1 = nn.Linear(1, num_hidden)

        torch.nn.init.xavier_uniform_(self.layer1.weight, gain=1.0)
        # torch.nn.init.xavier_uniform_(self.layer1.bias, gain=1.0)

        self.head = nn.Linear(num_hidden, 1)
        torch.nn.init.xavier_uniform_(self.head.weight, gain=1.0)
        # torch.nn.init.xavier_uniform_(self.head.bias, gain=1.0)

    def forward(self, x):
        l = torch.relu(self.layer1(x))
        l = self.head(l)
        return l


def distance(a,b):
    return torch.dist(a,b)
def train(model, x, y):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    loss.backward()

    optimizer = optim.Adam(model.parameters(), lr=step_size_single)
    # optimizer = optim.SGD(model.parameters(), lr=step_size)

    optimizer.step()
    optimizer.zero_grad()


error_single_over_runs = []
error_multi_over_runs = []

for r in range(num_runs):
    np.random.seed(r)
    error_list_single = []
    error_list_multi = []

    #initializing the models
    single_model = StateTransitionModel(num_hidden_units)
    multi_model = []
    for i in range(multi_model_num):
        multi_model.append(StateTransitionModel(num_hidden_units_multi//multi_model_num))


    #create the dataset
    x = np.random.uniform(range_data_points[0],range_data_points[1], num_data_points)
    x = np.reshape(np.sort(x), (num_data_points,1))
    y = x + np.sin(4*x) +np.sin(13*x)

    #train
    for e in tqdm(range(num_epochs)):
        #train single model
        for i in range(num_data_points // batch_size_single):
            ind = np.random.choice(num_data_points, batch_size_single)
            batch_x, batch_y = x[ind], y[ind]
            train(single_model, batch_x, batch_y)

        #train multi model
        for i in range(num_data_points // batch_size_multi):
            ind = np.random.choice(num_data_points, batch_size_multi)
            batch_x, batch_y = x[ind], y[ind]
            dist_list = []

            #find which model should be trained
            for c, mod in enumerate(multi_model):
                with torch.no_grad():
                    y_hat = mod(torch.from_numpy(batch_x).float())
                dist_list.append(distance(torch.from_numpy(batch_y).float(), y_hat))
            m = np.argmin(dist_list)
            train(multi_model[m], batch_x, batch_y)

        # draw plot till now
        if e % plot_show_epoch_freq == 0:
            with torch.no_grad():
                y_hat_single = single_model(torch.from_numpy(x).float())

                y_hat_each = []
                for c, mod in enumerate(multi_model):
                    y_hat_each.append( mod(torch.from_numpy(x).float()))

                y_hat_multi = []
                for i in range(num_data_points):
                    dif = []
                    for y_hat in y_hat_each:
                        dif.append(distance(y_hat[i], torch.from_numpy(y[i]).float()))
                    ind = np.argmin(dif)
                    y_hat_multi.append(y_hat_each[ind][i])
                y_hat_multi = np.asarray(y_hat_multi)
                y_hat_multi = np.reshape(y_hat_multi, (num_data_points, 1))

            # plt.title('after '+str(e+1)+' epochs')
            # plt.plot(x, y, 'g', label='ground truth')
            # plt.plot(x, y_hat_single, 'r', label='single model')
            # plt.plot(x, y_hat_multi, 'b', label='multi model')
            # plt.legend()
            # plt.show()

            distance_single = distance(torch.from_numpy(y).float(), y_hat_single)
            distance_multi = distance(torch.from_numpy(y).float(), torch.from_numpy(y_hat_multi).float())
            error_list_single.append(distance_single)
            error_list_multi.append(distance_multi)

    error_single_over_runs.append(error_list_single)
    error_multi_over_runs.append(error_list_multi)

    #test
    with torch.no_grad():
        y_hat_single = single_model(torch.from_numpy(x).float())

        y_hat_each = []
        for c, mod in enumerate(multi_model):
            y_hat_each.append( mod(torch.from_numpy(x).float()))

        y_hat_multi = []
        y_hat_multi2 = [[],[],[],[]]
        x_hat_multi = [[],[],[],[]]
        for i in range(num_data_points):
            dif = []
            for y_hat in y_hat_each:
                dif.append(distance(y_hat[i], torch.from_numpy(y[i]).float()))
            ind = np.argmin(dif)
            y_hat_multi.append(y_hat_each[ind][i])

            x_hat_multi[ind].append(x[i])
            y_hat_multi2[ind].append(y_hat_each[ind][i])


    plt.title('function')
    plt.plot(x, y, 'k', label='ground truth')
    plt.plot(x, y_hat_single, 'r', label='single model')
    plt.plot(x, y_hat_multi, 'b', label='multi model')
    plt.legend()
    plt.show()

    plt.plot(x, y, 'k', label='ground truth')
    plt.plot(x_hat_multi[0], y_hat_multi2[0],'c', label='0')
    plt.plot(x_hat_multi[1], y_hat_multi2[1],'m', label='1')
    plt.plot(x_hat_multi[2], y_hat_multi2[2],'y', label='2')
    plt.plot(x_hat_multi[3], y_hat_multi2[3],'g', label='3')

    plt.legend()
    plt.show()
    #
    # plt.title('model error')
    # plt.plot(error_list_single,'r', label='single model')
    # plt.plot(error_list_multi,'b', label='multi model')
    # plt.legend()
    # plt.show()


err_multi = np.asarray(error_multi_over_runs)
err_single = np.asarray(error_single_over_runs)

# with open('multi.npy', 'wb') as f:
#     np.save(f, err_multi)
# with open('single.npy', 'wb') as f:
#     np.save(f, err_single)

err_multi_mean = np.mean(err_multi, axis=0)
err_multi_bar = np.std(err_multi, axis=0)

err_single_mean = np.mean(err_single, axis=0)
err_single_bar = np.std(err_single, axis=0)

plt.plot(err_single_mean,'r', label='single model')
plt.fill_between(range(len(err_single_mean)),
                 err_single_mean - err_single_bar,
                 err_single_mean + err_single_bar,
                 facecolor='red', alpha=0.2, edgecolor='none')

plt.plot(err_multi_mean,'b', label='multi model')
plt.fill_between(range(len(err_multi_mean)),
                 err_multi_mean - err_multi_bar,
                 err_multi_mean + err_multi_bar,
                 facecolor='blue', alpha=0.2, edgecolor='none')

plt.legend()
plt.show()
