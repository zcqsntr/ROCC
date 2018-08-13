import numpy as np
import matplotlib.pyplot as plt


def plot_pops(xSol, save_path = False):
    '''Plots the time evolutions of all species in xSol'''

    xSol = np.array(xSol)

    num_species = len(xSol[0,:])
    T_MAX = len(xSol[:,0])

    for s in range(num_species):
        plt.plot(np.linspace(0,T_MAX,len(xSol[:,0])), xSol[:,s])

    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)

    try:
        plt.savefig(save_path)
        plt.xlabel('Timestep')
        plt.ylabel('Population (A.U.)')
        plt.title('Populations After Training')

        plt.close()
    except:
        pass


def plot_survival(ts, save_path, NUM_EPISODES, T_MAX, phase):
    '''Plots the time all species survived for against episode'''
    ts = ts[1:]

    plt.plot(np.linspace(0, NUM_EPISODES, len(ts)), ts)
    plt.ylim([-1, T_MAX+1])
    plt.xlim(xmin = 0)
    plt.xlabel('Episode')
    plt.title(phase + ' Survival Time')
    plt.ylabel('Timesteps Survived')
    plt.savefig(save_path)

    plt.close()


def plot_rewards(rewards, save_path, NUM_EPISODES, T_MAX, phase):
    '''Plots the total reward recieved per episode against episode'''
    
    plt.plot(np.linspace(0, NUM_EPISODES, len(rewards)), rewards)
    plt.xlim(xmin = 0)
    plt.title(phase + ' Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward Recieved')
    plt.savefig(save_path)

    plt.close()
