import numpy as np
import matplotlib.pyplot as plt
import Colab.utils as utils, Colab.config as config

with open("Colab/Logs/num_steps_run_list_MCTS.npy", 'rb') as f:
    num_steps_mcts = np.load(f)
with open("Colab/Logs/num_steps_run_list_DQN.npy", 'rb') as f:
    num_steps_dqn = np.load(f)
with open("Colab/Logs/num_steps_run_list_random.npy", 'rb') as f:
    num_steps_random = np.load(f)

print(num_steps_mcts.shape)
print(num_steps_dqn.shape)

mean_num_steps_dqn = np.mean(num_steps_dqn, axis=1)
var_num_steps_dqn = np.std(num_steps_dqn, axis=1)
mean_num_steps_mcts = np.mean(num_steps_mcts, axis=1)
var_num_steps_mcts = np.std(num_steps_mcts, axis=1)
mean_num_steps_random = np.mean(num_steps_random, axis=1)
var_num_steps_random = np.std(num_steps_random, axis=1)

ind = 0
plt.plot(mean_num_steps_dqn[ind],'r', label='dqn')
plt.fill_between(range(len(mean_num_steps_dqn[ind])),
                 mean_num_steps_dqn[ind] - var_num_steps_dqn[ind],
                 mean_num_steps_dqn[ind] + var_num_steps_dqn[ind],
                 facecolor='red', alpha=0.2, edgecolor='none')

plt.plot(mean_num_steps_mcts[ind],'g', label='mcts')
plt.fill_between(range(len(mean_num_steps_mcts[ind])),
                 mean_num_steps_mcts[ind] - var_num_steps_mcts[ind],
                 mean_num_steps_mcts[ind] + var_num_steps_mcts[ind],
                 facecolor='green', alpha=0.2, edgecolor='none')

plt.plot(mean_num_steps_random[ind],'b', label='random')
plt.fill_between(range(len(mean_num_steps_random[ind])),
                 mean_num_steps_random[ind] - var_num_steps_random[ind],
                 mean_num_steps_random[ind] + var_num_steps_random[ind],
                 facecolor='blue', alpha=0.2, edgecolor='none')
plt.legend()
plt.show()




with open("Colab/Logs/num_steps_run_list_DQNMCTS.npy", 'rb') as f:
    num_steps_mcts = np.load(f)
with open("Colab/Logs/sim_num_steps_run_list.npy", 'rb') as f:
    num_steps_dqn = np.load(f)
with open("Colab/Logs/consistency_num_steps_run_list.npy", 'rb') as f:
    consistency = np.load(f)


with open("Colab/Logs/num_steps_run_list_DQNMCTS2.npy", 'rb') as f:
    num_steps_mcts2 = np.load(f)
with open("Colab/Logs/sim_num_steps_run_list2.npy", 'rb') as f:
    num_steps_dqn2 = np.load(f)
with open("Colab/Logs/consistency_num_steps_run_list2.npy", 'rb') as f:
    consistency2 = np.load(f)
print(num_steps_mcts.shape)
print(num_steps_dqn.shape)

mean_num_steps_dqn = np.mean(num_steps_dqn, axis=1)
var_num_steps_dqn = np.std(num_steps_dqn, axis=1)
mean_num_steps_mcts = np.mean(num_steps_mcts, axis=1)
var_num_steps_mcts = np.std(num_steps_mcts, axis=1)
mean_num_steps_consistency = np.mean(consistency, axis=1)
var_num_steps_consistency = np.std(consistency, axis=1)

mean_num_steps_dqn2 = np.mean(num_steps_dqn2, axis=1)
var_num_steps_dqn2 = np.std(num_steps_dqn2, axis=1)
mean_num_steps_mcts2 = np.mean(num_steps_mcts2, axis=1)
var_num_steps_mcts2 = np.std(num_steps_mcts2, axis=1)
mean_num_steps_consistency2 = np.mean(consistency2, axis=1)
var_num_steps_consistency2 = np.std(consistency2, axis=1)
ind = 0
fig, axs = plt.subplots(2,2)

axs[0,0].title.set_text('DQN train with MCTS expansion')
axs[0,0].plot(mean_num_steps_dqn[ind],'r', label='dqn')
axs[0,0].fill_between(range(len(mean_num_steps_dqn[ind])),
                 mean_num_steps_dqn[ind] - var_num_steps_dqn[ind],
                 mean_num_steps_dqn[ind] + var_num_steps_dqn[ind],
                 facecolor='red', alpha=0.2, edgecolor='none')

axs[0,0].plot(mean_num_steps_mcts[ind],'g', label='mcts')
axs[0,0].fill_between(range(len(mean_num_steps_mcts[ind])),
                 mean_num_steps_mcts[ind] - var_num_steps_mcts[ind],
                 mean_num_steps_mcts[ind] + var_num_steps_mcts[ind],
                 facecolor='green', alpha=0.2, edgecolor='none')

axs[1,0].plot(mean_num_steps_consistency[ind],'b', label='consistency')
axs[1,0].fill_between(range(len(mean_num_steps_consistency[ind])),
                 mean_num_steps_consistency[ind] - var_num_steps_consistency[ind],
                 mean_num_steps_consistency[ind] + var_num_steps_consistency[ind],
                 facecolor='blue', alpha=0.2, edgecolor='none')

axs[0,1].title.set_text('DQN train with MCTS selection path')
axs[0,1].plot(mean_num_steps_dqn2[ind],'r', label='dqn')
axs[0,1].fill_between(range(len(mean_num_steps_dqn2[ind])),
                 mean_num_steps_dqn2[ind] - var_num_steps_dqn2[ind],
                 mean_num_steps_dqn2[ind] + var_num_steps_dqn2[ind],
                 facecolor='red', alpha=0.2, edgecolor='none')

axs[0,1].plot(mean_num_steps_mcts2[ind],'g', label='mcts')
axs[0,1].fill_between(range(len(mean_num_steps_mcts2[ind])),
                 mean_num_steps_mcts2[ind] - var_num_steps_mcts2[ind],
                 mean_num_steps_mcts2[ind] + var_num_steps_mcts2[ind],
                 facecolor='green', alpha=0.2, edgecolor='none')

axs[1,1].plot(mean_num_steps_consistency2[ind],'b', label='consistency')
axs[1,1].fill_between(range(len(mean_num_steps_consistency2[ind])),
                 mean_num_steps_consistency2[ind] - var_num_steps_consistency2[ind],
                 mean_num_steps_consistency2[ind] + var_num_steps_consistency2[ind],
                 facecolor='blue', alpha=0.2, edgecolor='none')
axs[0,0].legend()
axs[0,1].legend()
axs[1,0].legend()
axs[1,1].legend()
plt.show()































exit(0)
with open('multi.npy', 'rb') as f:
    err_multi = np.load(f)
with open('single.npy', 'rb') as f:
    err_single = np.load(f)

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

exit(0)
'''
model_error0.125.npy
inf 1
1936.8844121456145 2
1325.2853795826443 4

model_error0.0625.npy
406892021450.05316 1
1288.575113272667 2
793.5572925895451 4

model_error0.03125.npy
471.2412468723953 1
405.7960289604963 2
431.49920514076916 4

model_error0.015625.npy
97.64286422617738 1
127.7020316816867 2
125.99224670529378 4

model_error0.0078125.npy
37.364617464318854 1
39.84164921566847 2
36.88163374476133 4         ***********

model_error0.00390625.npy
39.412814384698876 1
35.81449481248855 2       ***********
43.6953097369522 4

model_error0.001953125.npy
33.828165540099164 1      ***********
41.80296202823518 2
45.13627097383143 4

model_error0.00048828125.npy
47.972808397561295 1
68.23368751034141 2
98.08690240979196 4

model_error0.000244140625.npy
58.19502714946866 1
102.43051464930173 2
138.10314898416394 4
'''

if False:
    color=['blue','orange','green']
    agent_list = ['forward', 'backward', 'backward+c']
    with open('num_steps_run_list.npy', 'rb') as f:
        res1 = np.load(f)
    with open('num_steps_run_list2.npy', 'rb') as f:
        res2 = np.load(f)
    with open('num_steps_run_list3.npy', 'rb') as f:
        res3 = np.load(f)

    res = res3#np.concatenate((res1),axis=1)
    print(res.shape)
    # for r in range(res.shape[1]):
    #     for a in range(res.shape[0]):
    #         agent_name = config.agent_list[a].name
    #         utils.draw_plot(range(len(res[a, r])), res[a, r],
    #                         xlabel='episode_num', ylabel='num_steps', show=False,
    #                         label=agent_name, title='run_number ' + str(r + 1))
    #     plt.show()


    for agent_num in range(3):
        plt.plot(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
                       label=agent_list[agent_num], color=color[agent_num])
        AUC = sum(np.mean(res[agent_num], axis=0))
        print(AUC, agent_list[agent_num])
    plt.legend()
    plt.savefig('results')



    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    for x in range(axs.shape[0]):
        for y in range(axs.shape[1]):
            axs[x,y].set_xlabel('episode_num')
            axs[x,y].set_ylabel('num_steps')
    # *********
    agent_num = 0
    i,j = 0,0
    axs[i,j].errorbar(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
                      yerr= np.std(res[agent_num], axis=0), label=agent_list[agent_num], color=color[agent_num])
    axs[i,j].legend()

    # *********
    agent_num = 1
    i,j = 0,1
    axs[i,j].errorbar(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
                      yerr= np.std(res[agent_num], axis=0), label=agent_list[agent_num], color=color[agent_num])
    axs[i,j].legend()

    # *********
    agent_num = 2
    i,j = 1,0
    axs[i,j].errorbar(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
                      yerr= np.std(res[agent_num], axis=0), label=agent_list[agent_num], color=color[agent_num])
    axs[i,j].legend()

    # *********
    for agent_num in range(3):
        i,j = 1,1
        axs[i,j].errorbar(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
                          yerr= np.std(res[agent_num], axis=0), label=agent_list[agent_num], color=color[agent_num])
        axs[i,j].legend()

    plt.savefig('results+errorbar')
    plt.show()

if True:
    color=['blue','red','green']
    agent_list = ['1', '2', '4']

    with open('model_error0.001953125.npy', 'rb') as f:
        model_error1 = np.load(f)
        model_error_samples1 = np.load(f)

    with open('model_error0.00390625.npy', 'rb') as f:
        model_error2 = np.load(f)
        model_error_samples2 = np.load(f)

    with open('model_error0.0078125.npy', 'rb') as f:
        model_error3 = np.load(f)
        model_error_samples3 = np.load(f)

    with open('model_error_run4.npy','rb') as f:
        model_error = np.load(f)
        model_error_samples = np.load(f)

    # print(model_error)
    # print(model_error_samples)

    # Each Run Plot
    # for r in range(model_error.shape[1]):
    #     for a in range(model_error.shape[0]):
    #         agent_name = config.agent_list[a].name
    #         utils.draw_plot(model_error_samples[a, r], model_error[a, r],
    #                         xlabel='num_samples', ylabel='model_error', show=False,
    #                         label=agent_name, title='run_number ' + str(r + 1))
    #     plt.show()


    # AUC
    # for agent_num in range(len(agent_list)):
    #     x = np.mean(model_error_samples[agent_num], axis=0)
    #     y = np.mean(model_error[agent_num], axis=0)
    #     plt.plot(x, y, label=config.agent_list[agent_num].name + agent_list[agent_num],
    #              color=color[agent_num])
    #     AUC = sum(np.mean(model_error[agent_num], axis=0))
    #     print(AUC, agent_list[agent_num])
    # plt.legend()
    # plt.show()


    # Error Bar
    for agent_num in range(len(agent_list)):
        # if agent_num == 0:
        #     model_error_samples = model_error_samples1
        #     model_error = model_error1
        # elif agent_num == 1:
        #     model_error_samples = model_error_samples2
        #     model_error = model_error2
        # elif agent_num == 2:
        #     model_error_samples = model_error_samples3
        #     model_error = model_error3

        x = np.mean(model_error_samples[agent_num], axis=0)
        y = np.mean(model_error[agent_num], axis=0)
        yerr = np.std(model_error[agent_num], axis=0)
        # plt.errorbar(x, y, yerr= yerr,
        #          label=agent_list[agent_num] + agent_list[agent_num],
        #          color=color[agent_num])
        plt.plot(x, y, label=config.agent_list[agent_num].name + agent_list[agent_num],
                              color=color[agent_num])
        plt.fill_between(x, y - yerr, y + yerr, facecolor=color[agent_num], alpha=0.2, edgecolor='none')

    plt.legend()
    plt.show()
    plt.savefig('model_error+errorbar')



    # f, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    # for x in range(axs.shape[0]):
    #     for y in range(axs.shape[1]):
    #         axs[x,y].set_xlabel('episode_num')
    #         axs[x,y].set_ylabel('num_steps')
    # # *********
    # agent_num = 0
    # i,j = 0,0
    # axs[i,j].errorbar(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
    #                   yerr= np.std(res[agent_num], axis=0), label=agent_list[agent_num], color=color[agent_num])
    # axs[i,j].legend()
    #
    # # *********
    # agent_num = 1
    # i,j = 0,1
    # axs[i,j].errorbar(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
    #                   yerr= np.std(res[agent_num], axis=0), label=agent_list[agent_num], color=color[agent_num])
    # axs[i,j].legend()
    #
    # # *********
    # agent_num = 2
    # i,j = 1,0
    # axs[i,j].errorbar(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
    #                   yerr= np.std(res[agent_num], axis=0), label=agent_list[agent_num], color=color[agent_num])
    # axs[i,j].legend()
    #
    # # *********
    # for agent_num in range(3):
    #     i,j = 1,1
    #     axs[i,j].errorbar(range(len(np.mean(res[agent_num], axis=0))), np.mean(res[agent_num], axis=0),
    #                       yerr= np.std(res[agent_num], axis=0), label=agent_list[agent_num], color=color[agent_num])
    #     axs[i,j].legend()
    #
    # plt.savefig('results+errorbar')
    # plt.show()








