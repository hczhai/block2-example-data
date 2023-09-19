
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

molx = matplotlib.image.imread('fe2sch3.png')
molx[:, :, 3][np.sum(molx[:, :, :3], axis=-1) / 3 > 0.9] = 0
imagebox = OffsetImage(molx, zoom=0.175)

matplotlib.rc('font', size=14)
figure(figsize=(12 / 2, 9 / 2), dpi=150)

f, axs = plt.subplots(1, 1, sharey=True)
f.set_size_inches(w=12 / 2, h=9 / 2)

bdims = []
times = []
bdimsx = []
timesx = []

with open("../00-HC/01-dmrg-nsa-rev.out") as f:
    for l in f.readlines():
        if "Bond dimension =" in l:
            bdims.append(int(l.split()[11]))
        elif "Time sweep =" in l:
            times.append(float(l.split()[3]))

with open("../00-HC/00-dmrg-rev.out") as f:
    for l in f.readlines():
        if "Bond dimension =" in l:
            bdimsx.append(int(l.split()[11]))
        elif "Time sweep =" in l:
            timesx.append(float(l.split()[3]))

bdims = bdims[3::4][::-1]
times = times[3::4][::-1]
bdimsx = bdimsx[3::4][::-1]
timesx = timesx[3::4][::-1]

import scipy.stats
reg = scipy.stats.linregress(np.log(bdims[-3:]), np.log(times[-3:]))
regx = scipy.stats.linregress(np.log(bdimsx[-3:]), np.log(timesx[-3:]))
x_reg = np.array([900, 6000])
print(reg.slope)
print(regx.slope)

# https://coolors.co/a8cbf0-4e4187-eed6bf-cbe1ce-8eb8a7
xcolors = ["#EED6BF", "#CBE1CE", "#7A7A7A", "#A8CBF0"]
mcolors = ["#DDAE7E", "#7FB685", "#2E2E2E", "#3083DC"]
marker = ['o', '^', 's', 'p']

plt.grid(which='major', axis='both', alpha=0.5)
plt.plot(x_reg, np.exp(reg.intercept + reg.slope * np.log(x_reg)), '--', linewidth=1.5, color=mcolors[2],
    label='Fit $T = M^{%.2f} /\\ %.0f $' % (reg.slope, 1 / np.exp(reg.intercept)))
plt.plot(bdims, times, marker=marker[2], mfc='white', mec=mcolors[2], color=mcolors[2], linewidth=0, markersize=7, label='DMRG (non-spin-adapted)')
plt.plot(x_reg, np.exp(regx.intercept + regx.slope * np.log(x_reg)), '-.', linewidth=1.5, color=mcolors[3],
    label='Fit $T = M^{%.2f} /\\ %.0f $' % (regx.slope, 1 / np.exp(regx.intercept)))
plt.plot(bdimsx, timesx, marker=marker[3], mfc='white', mec=mcolors[3], color=mcolors[3], linewidth=0, markersize=7, label='DMRG (spin-adapted)')
plt.xlim(900, 5999)
plt.xscale('log')
plt.yscale('log')
plt.xticks(bdims, list(map(str, bdims)))
axs.set_ylabel("Wall time per sweep $T$ (seconds)")
axs.set_xlabel("MPS bond dimension $M$")

axs.add_artist(AnnotationBbox(imagebox, (3400, 220), frameon=False))
axs.text(2970, 65, '$[\mathrm{Fe_2S(CH_3)(SCH_3)_4}]^{3-}$', color='#000000', fontsize=12)
axs.legend(loc='upper left', fontsize=12)
plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.14, wspace=0.0, hspace=0.0)
plt.savefig('fig-01.png', dpi=600)
