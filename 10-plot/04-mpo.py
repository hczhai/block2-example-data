
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

molx = matplotlib.image.imread('femoco.png')
molx[:, :, :3][molx[:, :, :3] < 100] = molx[:, :, :3][molx[:, :, :3] < 100] - 50
molx[:, :, :3][molx[:, :, :3] < 0] = 0
imagebox = OffsetImage(molx, zoom=0.13)

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

# https://coolors.co/a8cbf0-4e4187-eed6bf-cbe1ce-8eb8a7
xcolors = ["#EED6BF", "#CBE1CE", "#7A7A7A", "#A8CBF0"]
mcolors = ["#DDAE7E", "#7FB685", "#2E2E2E", "#3083DC"]
marker = ['o', '^', 's', 'p']

plt.grid(which='major', axis='both', alpha=0.5)
plt.plot((1E-13, 1E-4), (bip_bdim, bip_bdim), '-.', color=mcolors[3], linewidth=1.5, label='Bipartite approach')
plt.plot(cutoffs, bdims, '--', marker=marker[2], mfc='white', mec=mcolors[2], color=mcolors[2], linewidth=1.5, markersize=7, label='SVD approach')
plt.xlim(5E-13, 2E-5)
plt.xscale('log')
# plt.yscale('log')
plt.xticks(cutoffs[::2], ["%.0E" % x for x in cutoffs[::2]])
axs.set_ylabel("MPO bond dimension")
axs.set_xlabel("MPO singular value cutoff")

axs.add_artist(AnnotationBbox(imagebox, (1E-6, 3500), frameon=False))
axs.legend(loc='lower left', fontsize=12)
plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.14, wspace=0.0, hspace=0.0)
plt.savefig('fig-04.pdf', dpi=600)
