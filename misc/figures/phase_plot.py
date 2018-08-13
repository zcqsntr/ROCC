

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

matplotlib.rcParams.update({'font.size': 20})


pops = np.load('/Users/Neythen/masters_project/results/lookup_table_results/smaller_target_fixed/WORKING/WORKING_data/LTPops.npy')

plt.figure(figsize = (16, 13))
plt.plot(np.linspace(0,400, 400), pops[0:400,0])
plt.plot(np.linspace(0,400, 400), pops[0:400,1])
plt.xlabel('Timestep')
plt.ylabel('Population')

plt.figure(figsize = (16, 13))
plt.plot(pops[0:500,0], pops[0:500, 1], 'black')
plt.xlim([0, 1000])
plt.ylim([0, 1000])
plt.savefig('phase.png', transparent = True)
plt.xlabel('N1')
plt.ylabel('N2')
