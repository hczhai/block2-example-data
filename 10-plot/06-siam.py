
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import matplotlib

matplotlib.rc('font', size=14)
figure(figsize=(12 / 2, 9 / 2), dpi=150)

f, axs = plt.subplots(1, 1, sharey=True)
f.set_size_inches(w=12 / 2, h=9 / 2)

tarr = []
jarr = []

with open("../06-SIAM/00-dmrg.out") as f:
    for l in f.readlines():
        if "Norm =" in l:
            tarr.append(float(l.split()[2]))
            jarr.append(float(l.split()[-4]))

# https://coolors.co/a8cbf0-4e4187-eed6bf-cbe1ce-8eb8a7
xcolors = ["#EED6BF", "#CBE1CE", "#7A7A7A", "#A8CBF0"]
mcolors = ["#DDAE7E", "#7FB685", "#2E2E2E", "#3083DC"]
marker = ['o', '^', 's', 'p']

plt.grid(which='major', axis='both', alpha=0.5)
plt.plot(tarr, jarr, '--', color=mcolors[2], linewidth=1.5, markersize=7, label='Time-dependent DMRG')
plt.xlim(0, max(tarr))
plt.ylim(0, 1.4)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(bdims, list(map(str, bdims)))
axs.set_ylabel("Current $J(t) / V$")
axs.set_xlabel("Time $t$")

axs.text(5, 0.18, '$\\hat{H} = \\hat{H}_{\\mathrm{dot}} + \\hat{H}_{\\mathrm{leads}}$', color='#000000', fontsize=14, zorder=999)
axs.text(5, 0.08, 'Single-impurity Anderson model', color='#000000', fontsize=14, zorder=999)
axs.legend(loc='upper right', fontsize=12)
plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.14, wspace=0.0, hspace=0.0)
plt.savefig('fig-06.png', dpi=600)
