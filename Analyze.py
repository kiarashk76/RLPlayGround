import numpy as np
import matplotlib.pyplot as plt
import Colab.utils as utils, Colab.config as config

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








