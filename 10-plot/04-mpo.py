
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib

molx = matplotlib.image.imread('femoco.png')
molx[:, :, :3][molx[:, :, :3] < 100] = molx[:, :, :3][molx[:, :, :3] < 100] - 50
molx[:, :, :3][molx[:, :, :3] < 0] = 0
imagebox = OffsetImage(molx, zoom=0.08)

matplotlib.rc('font', size=14)
figure(figsize=(12 / 2, 9 / 2), dpi=150)

f, axs = plt.subplots(1, 1, sharey=True)
f.set_size_inches(w=12 / 2, h=9 / 2)

cutoffs = []
bdims = []
conv_bdim, bip_bdim = 0, 0

with open("../04-FeMoco/00-dmrg.out") as f:
    for l in f.readlines():
        if "FastBlockedSVD | Cutoff =" in l:
            cutoffs.append(float(l.split()[17]))
        elif "FastBipartite bond dimension" in l:
            bip_bdim = int(l.split()[13])
        elif "FastBlockedSVD bond dimension" in l:
            bdims.append(int(l.split()[13]))

acc_cuts = []
acc_eners = []

with open("../04-FeMoco/02-dmrg.out") as f:
    p = -1
    for l in f.readlines():
        if "Time init sweep =" in l:
            p = 0
        elif p == 2:
            x, y = [float(x) for x in l.strip().split()]
            acc_cuts.append(x)
            acc_eners.append(y)
            p = -1
        elif p >= 0:
            p += 1

acc_cuts = acc_cuts[:-1]
acc_eners = [abs(x - acc_eners[-1]) for x in acc_eners[:-1]]

# https://coolors.co/a8cbf0-4e4187-eed6bf-cbe1ce-8eb8a7
xcolors = ["#EED6BF", "#CBE1CE", "#7A7A7A", "#A8CBF0"]
mcolors = ["#DDAE7E", "#7FB685", "#2E2E2E", "#3083DC"]
marker = ['o', '^', 's', 'p']

plt.grid(which='major', axis='both', alpha=0.5)
plt.plot((np.sqrt(1E-13), 1E-2), (bip_bdim, bip_bdim), '-.', color=mcolors[3], linewidth=1.5, label='Bipartite approach')
plt.plot(np.sqrt(cutoffs), bdims, '--', marker=marker[2], mfc='white', mec=mcolors[2], color=mcolors[2], linewidth=1.5, markersize=7, label='SVD approach')
plt.xlim(np.sqrt(5E-13), np.sqrt(2E-5))
plt.xscale('log')
# plt.yscale('log')
plt.xticks([1E-6, 1E-5, 1E-4, 1E-3], ["$10^{-%s}$" % (("%.E" % x).split('-')[-1].split('0')[-1]) for x in [1E-6, 1E-5, 1E-4, 1E-3]])
axs.set_ylabel("MPO bond dimension")
axs.set_xlabel("MPO singular value cutoff")

axins = inset_axes(plt.gca(), width="50%", height="48%", bbox_to_anchor=(0.15, 0.10, 0.84, 0.84), bbox_transform=plt.gca().transAxes, loc='upper right')
axins.grid(which='major', axis='both', alpha=0.5)
axins.plot(np.sqrt(acc_cuts), acc_eners, '--', marker=marker[2], mfc='white', mec=mcolors[2], color=mcolors[2], linewidth=1.5, markersize=7, label='SVD approach')
axins.set_xlim(np.sqrt(5E-13), np.sqrt(2E-5))
axins.set_ylim(1E-9, 1E-1)
axins.set_xscale('log')
axins.set_yscale('log')
axins.set_xticks([1E-6, 1E-5, 1E-4, 1E-3], ["$10^{-%s}$" % (("%.E" % x).split('-')[-1].split('0')[-1]) for x in [1E-6, 1E-5, 1E-4, 1E-3]])
axins.set_yticks([1E-8, 1E-5, 1E-2], ["$10^{-%s}$" % (("%.E" % x).split('-')[-1].split('0')[-1]) for x in [1E-8, 1E-5, 1E-2]])
axins.tick_params(labelsize=12)
axins.set_ylabel("$\\langle \\hat{H} \\rangle$ error (Ha)", fontsize=12)
axins.set_xlabel("MPO singular value cutoff", fontsize=12)

axs.add_artist(AnnotationBbox(imagebox, (5E-12, 2800), frameon=False))
axs.legend(loc='lower left', fontsize=12)
plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.14, wspace=0.0, hspace=0.0)
plt.savefig('fig-04.pdf', dpi=600)
