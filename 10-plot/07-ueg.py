
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import matplotlib

matplotlib.rc('font', size=14)
figure(figsize=(12 / 2, 9 / 2), dpi=150)

f, axs = plt.subplots(1, 1, sharey=True)
f.set_size_inches(w=12 / 2, h=9 / 2)

betas = []
eners = []

with open("../07-UEG/00-dmrg.out") as f:
    for l in f.readlines():
        if "Norm =" in l:
            betas.append(float(l.split()[2]))
            eners.append(float(l.split()[8]))

betas = np.array(betas[1:])
eners = eners[1:]

# https://coolors.co/a8cbf0-4e4187-eed6bf-cbe1ce-8eb8a7
xcolors = ["#EED6BF", "#CBE1CE", "#7A7A7A", "#A8CBF0"]
mcolors = ["#DDAE7E", "#7FB685", "#2E2E2E", "#3083DC"]
marker = ['o', '^', 's', 'p']

plt.grid(which='major', axis='both', alpha=0.5)
plt.plot(1 / betas, eners, '--', marker=marker[0], mfc='white', mec=mcolors[2],
    linewidth=1.5, markersize=4, color=mcolors[2], label='Finite temperature DMRG (ancilla approach)')
plt.xlim(0, 1 / min(betas))
plt.ylim(0, 9)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(bdims, list(map(str, bdims)))
axs.set_ylabel("Electronic energy $E(T)$")
axs.set_xlabel("Temperature $T$")

axs.text(5, 1.5, '$\\hat{H} = \\sum_{\\mathbf{k}} \\frac{|\\mathbf{k}|^2}{2}\\hat{a}^\\dagger_{\\mathbf{k}}\\hat{a}_{\\mathbf{k}}$' +
    '$\\ +\\ \\frac{1}{2\\Omega} \\sum_{\\mathbf{k}\\neq \\mathbf{0},\\mathbf{k}_1,\\mathbf{k}_2}$' +
    '$\\hat{a}^\\dagger_{\\mathbf{k}_1+\\mathbf{k}}\\hat{a}^\\dagger_{\\mathbf{k}_2-\\mathbf{k}}\\hat{a}_{\\mathbf{k}_2}\\hat{a}_{\\mathbf{k}_1}$', color='#000000', fontsize=14, zorder=999)
axs.text(5, 0.5, '3D Uniform Electron Gas', color='#000000', fontsize=14, zorder=999)
axs.legend(loc='upper left', fontsize=12)
plt.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.14, wspace=0.0, hspace=0.0)
plt.savefig('fig-07.png', dpi=600)
