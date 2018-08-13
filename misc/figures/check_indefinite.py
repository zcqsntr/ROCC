import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

from plot_funcs import *
from utilities import *



sing_aux_path_LT = '/Users/Neythen/masters_project/results/lookup_table_results/single_aux_repeats/repeat0/WORKING_data/LTPops.npy'
sing_aux_path_NN = '/Users/Neythen/masters_project/results/Q_learning_results/single_aux_repeats/repeat1/WORKING_data/QPops.npy'

doub_aux_path_LT = '/Users/Neythen/masters_project/results/lookup_table_results/auxotroph_section_10_repeats/WORKING/repeat3/WORKING_data/LTPops.npy'
doub_aux_path_NN = '/Users/Neythen/masters_project/results/Q_learning_results/double_auxotroph_repeats/WORKING/repeat3/WORKING_data/QPops.npy'


path = sing_aux_path_NN
pops = np.load(path)[100:]


pop1 = pops[:, 0]
pop2 = pops[:, 1]

print(max(pop1))
print(max(pop2))

indices1 = [i for i, x in enumerate(pop1) if x >= max(pop1)-0.0001]
indices2 = [i for i, x in enumerate(pop2) if x >= max(pop2)-0.0001]

values1 = [pop1[i] for i in range(1,len(pop1)-1) if pop1[i-1] < pop1[i] and pop1[i+1] < pop1[i]]
values2 = [pop2[i] for i in range(1,len(pop2)-1) if pop2[i-1] < pop2[i] and pop2[i+1] < pop2[i]]

plt.plot(values1)
plt.figure()
plt.plot(values2)
plt.show()
