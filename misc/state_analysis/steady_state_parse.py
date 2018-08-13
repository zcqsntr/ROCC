import numpy as np
from utilities import *
#f = open('/Users/Neythen/masters_project/results/lookup_table_results/use_for_auxotroph_section/steady_state_sim')
f = open('/Users/Neythen/masters_project/results/lookup_table_results/use_for_auxotroph_section/steady_state_sim')

C0 = 1.
q = 0.5


# read in steady states
for line in f:
    if line[0] == '[':
        line = line.replace("[", " ")
        line = line.replace("]", " ")
        line = line.split()
        line = [float(l) for l in line]
        N1 = line[0]
        N2 = line[1]
        C1 = line[2]
        C2 = line[3]

        u1 = monod2(C1, C0, 2, 0.00049,0.00006845928)
        u2 = monod2(C2, C0, 2, 0.00000102115,0.00006845928)
        print(u1, u2)
