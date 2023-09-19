
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

molx = matplotlib.image.imread('dycl6.png')
molx[:, :, 3][molx[:, :, 0] > 0.9] = 0
imagebox = OffsetImage(molx, zoom=0.2)

matplotlib.rc('font', size=14)
figure(figsize=(12 / 2, 9 / 2), dpi=150)

f, axs = plt.subplots(1, 1, sharey=True)
f.set_size_inches(w=12 / 2, h=9 / 2)

mono_x    = [189, 273, 441]
mono_emin = np.array([21.2978, 33.5748, 21.1683, 21.3558])
mono_emax = np.array([254.6367, 338.2681, 251.9965, 254.8903])
mono_emin = np.abs(mono_emin - mono_emin[0])[1:]
mono_emax = np.abs(mono_emax - mono_emax[0])[1:]

marker = ['^', 'o', 'p', '*']
mcolor = ["#FF6F59", "#2E2E2E", "#7A7A7A", "#3083DC"]

plt.grid(which='major', axis='both', alpha=0.5)
plt.plot(mono_x, mono_emin,
    '--', markersize=7, linewidth=1.5, marker=marker[0], mec=mcolor[1], mfc='white', color=mcolor[1],
    label="2nd excited state (with SOC)")
plt.plot(mono_x, mono_emax,
    '-.', markersize=7, linewidth=1.5, marker=marker[2], mec=mcolor[3], mfc='white', color=mcolor[3],
    label="15th excited state (with SOC)")

plt.yscale('log')
plt.xticks(mono_x, [str(x) for x in mono_x])
plt.ylim((3E-2, 5E2))
plt.xlim((160, 480))
plt.ylabel("$\\left|E_{\\mathrm{2\u2010step}} - E_{\\mathrm{1\u2010step}}\\right|\\ \\ (\\mathrm{cm}^{-1})$")
plt.xlabel("Number of 2-step pure-spin states")

axs.add_artist(AnnotationBbox(imagebox, (405, 8), frameon=False))
axs.text(425, 2.0, '$[\mathrm{DyCl_6}]^{3-}$', color='#000000', fontsize=12)
axs.legend(loc='upper right', fontsize=12)
plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.14, wspace=0.0, hspace=0.0)
plt.savefig('fig-02.png', dpi=600)
