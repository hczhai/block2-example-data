
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

molx = matplotlib.image.imread('fe2ocl6.png')
molx[:, :, 3][np.sum(molx[:, :, :3], axis=-1) / 3 > 0.9] = 0
imagebox = OffsetImage(molx, zoom=0.2)

matplotlib.rc('font', size=14)
figure(figsize=(12 / 2, 9 / 2), dpi=150)

f, axs = plt.subplots(1, 1, sharey=True)
f.set_size_inches(w=12 / 2, h=9 / 2)

for idx in [5, 6]:

    dws = []
    bdims = []
    eners = []
    times = []

    with open("../05-Fe2OCl6/runs/stdmrg-%d/hife.out.5" % idx) as f:
        ok = False
        for l in f.readlines():
            if "EST(T) =" in l:
                ok = True
            elif ok and "DW" in l:
                eners.append(float(l.split()[7]))
                dws.append(float(l.split()[-1]))
            elif ok and "Bond dimension =" in l:
                bdims.append(int(l.split()[11]))
            elif ok and "Time sweep =" in l:
                times.append(float(l.split()[3]))

    bdims = bdims[3::4][::-1]
    times = times[3::4][::-1]
    eners = eners[3::4][::-1]
    dws = dws[3::4][::-1]

    if idx == 5:
        dws5 = dws[2:-1][::-1]
        eners5 = eners[2:-1][::-1]
    else:
        dws6 = dws[2:-1][::-1]
        eners6 = eners[2:-1][::-1]

import scipy.stats

reg5 = scipy.stats.linregress(dws5, eners5)
emin5, emax5 = min(eners5), max(eners5)
print('DMRG energy (extrapolated-5) = %20.15f +/- %15.10f' %
      (reg5.intercept, abs(reg5.intercept - emin5) / 5))

reg6 = scipy.stats.linregress(dws6, eners6)
emin6, emax6 = min(eners6), max(eners6)
print('DMRG energy (extrapolated-6) = %20.15f +/- %15.10f' %
      (reg6.intercept, abs(reg6.intercept - emin6) / 5))

eners5 = np.array(eners5) - reg5.intercept
eners6 = np.array(eners6) - reg6.intercept

# https://coolors.co/a8cbf0-4e4187-eed6bf-cbe1ce-8eb8a7
xcolors = ["#EED6BF", "#CBE1CE", "#7A7A7A", "#A8CBF0"]
mcolors = ["#DDAE7E", "#7FB685", "#2E2E2E", "#3083DC"]
marker = ['o', '^', 's', 'p']

au2cm = 219474.631115585274529

import matplotlib.pyplot as plt
from matplotlib import ticker
de5 = emax5 - emin5
de6 = emax6 - emin6
x_reg = np.array([0, dws5[-1] + dws5[0] + dws6[-1] + dws6[0]])
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1E}"))
plt.plot(x_reg, reg5.slope * x_reg * au2cm, '--', linewidth=1, color=mcolors[2])
plt.plot(x_reg, reg6.slope * x_reg * au2cm, '--', linewidth=1, color=mcolors[3])
plt.plot(dws5, eners5 * au2cm, ' ', marker='s', mfc='white', mec=mcolors[2], color=mcolors[2], markersize=7)
plt.plot(dws6, eners6 * au2cm, ' ', marker='p', mfc='white', mec=mcolors[3], color=mcolors[3], markersize=7)

plt.plot([-1], [-1], '--', marker='s', mfc='white', mec=mcolors[2], color=mcolors[2], markersize=7, linewidth=1, label='DMRG for $\\mathrm{e}^{-\\hat{T}} \\hat{H} \\mathrm{e}^{\\hat{T}}$ ($S_z = 5$)')
plt.plot([-1], [-1], '--', marker='p', mfc='white', mec=mcolors[3], color=mcolors[3], markersize=7, linewidth=1, label='DMRG for $\\mathrm{e}^{-\\hat{T}} \\hat{H} \\mathrm{e}^{\\hat{T}}$ ($S_z = 0$)')
plt.xlim((0, 9E-7))
plt.ylim((-de6 * 0.1 * au2cm, (emax6 + de6 * 0.2 - reg6.intercept) * au2cm))
plt.xlabel("Largest discarded weight")
plt.ylabel("Energy error $E - E_{\\mathrm{extrap}}$ ($\\mathrm{cm}^{-1}$)")

axs.text(2E-8, 24, 'Estimated $J_{\\mathrm{CCSD/ST\u2010DMRG}} = %.0f\\ \\mathrm{cm}^{-1}$' % ((reg6.intercept - reg5.intercept) / (5 * (5 + 1)) * au2cm), color='#000000', fontsize=12)

plt.xticks([0, 2E-7, 4E-7, 6E-7, 8E-7], list(["%.0E" % x for x in [0, 2E-7, 4E-7, 6E-7, 8E-7]]))
axs.add_artist(AnnotationBbox(imagebox, (7E-7, 7), frameon=False))
axs.text(6.5E-7, 0, '$[\mathrm{Fe_2OCl_6}]^{2-}$', color='#000000', fontsize=12)
axs.legend(loc='upper left', fontsize=12)
plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.14, wspace=0.0, hspace=0.0)
plt.savefig('fig-05.png', dpi=600)
