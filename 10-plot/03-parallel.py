
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

molx = matplotlib.image.imread('benzene.png')
molx[:, :, 3][np.sum(molx[:, :, :3], axis=-1) / 3 > 0.9] = 0
imagebox = OffsetImage(molx, zoom=0.16)

matplotlib.rc('font', size=14)
figure(figsize=(12 / 2, 9 / 2), dpi=150)

f, axs = plt.subplots(1, 1, sharey=True)
f.set_size_inches(w=12 / 2, h=9 / 2)

m2500 = [
    (2800, 816.4305),
    (1960, 893.845),
    (1512, 1486.5785),
    (1008, 1528.5325),
    (672, 1855.03225),
    (448, 3144.81875)
]

m3000 = [
    (2800, 1105.048),
    (1960, 1290.64275),
    (1512, 2048.795),
    (1008, 2107.1565),
    (672, 2541.85125),
    (448, 5252.682)
]

m3500 = [
    (1960, 2196.99075),
    (1512, 3191.51625),
    (1008, 3496.46275),
    (672, 4380.087),
    (448, 7639.537)
]

m4000 = [
    (2800, 2539.2995),
    (1960, 3051.2635),
    (1512, 4378.5395),
    (1008, 5079.4765),
    (672, 6158.32725),
    (448, 12740.41875)
]

m5000 = [
    (2800, 4525.79975),
    (1960, 5317.447),
    (672, 11334.841),
    (448, 22211.886)
]

m6000 = [
    (2800, 7419.294),
    (1960, 8696.30275),
    (448, 35450.86625)
]

plt.grid(which='major', axis='both', alpha=0.5)

marker = ['^', 'o', 'p', '*']
mcolor = ["#FF6F59", "#2E2E2E", "#7A7A7A", "#3083DC"]

plt.plot((448, 2800), (1, 1 * 2800 / 448), '--', color=mcolor[1], linewidth=1.5, label='ideal')
plt.plot([x[0] for x in m3000], [m3000[-1][1] / x[1] for x in m3000], markersize=7,
    linewidth=0, marker=marker[2], mec=mcolor[1], mfc='white', color=mcolor[1], label='MPS bond dimension $M$ = 3000')
plt.plot([x[0] for x in m4000], [m4000[-1][1] / x[1] for x in m4000], markersize=7,
    linewidth=0, marker=marker[0], mec=mcolor[3], mfc='white', color=mcolor[3], label='MPS bond dimension $M$ = 4000')

plt.xlim((420, 3000))
plt.ylim((0.9, 6.5))
plt.yscale('log')
plt.xscale('log')
plt.yticks([1, 2, 3, 4, 5, 6], list(map(str, [1, 2, 3, 4, 5, 6])))
plt.xticks([448, 672, 1008, 1512, 1960, 2800], list(map(str, [448, 672, 1008, 1512, 1960, 2800])))
plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.legend(loc='lower right', fontsize=9)
plt.xlabel('Number of CPU cores')
plt.ylabel('Average speed-up relative to 448 cores')

axs.add_artist(AnnotationBbox(imagebox, (2000, 1.6), frameon=False))
axs.legend(loc='upper left', fontsize=12)
plt.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.14, wspace=0.0, hspace=0.0)
plt.savefig('fig-03.png', dpi=600)