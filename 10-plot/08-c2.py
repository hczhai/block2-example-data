
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

molx = matplotlib.image.imread('c2.png')
molx[:, :, 3][np.sum(molx[:, :, :3], axis=-1) / 3 > 0.9] = 0
imagebox = OffsetImage(molx, zoom=0.2)

matplotlib.rc('font', size=14)
figure(figsize=(12 / 2, 9 / 2), dpi=150)

f, axs = plt.subplots(2, 1, sharex=True)
f.set_size_inches(w=12 / 2, h=18 / 2)

dists = np.arange(1.80, 4.16, 0.05)

ref_fci_sa = np.array([
    -75.4549, -75.5132, -75.5621, -75.6026, -75.6358, -75.6628, -75.6843, -75.7010, -75.7136, -75.7227,
    -75.7287, -75.7320, -75.7332, -75.7324, -75.7300, -75.7263, -75.7215, -75.7159, -75.7095, -75.7026,
    -75.6953, -75.6878, -75.6802, -75.6726, -75.6652, -75.6580, -75.6512, -75.6467, -75.6420, -75.6373,
    -75.6326, -75.6278, -75.6230, -75.6183, -75.6137, -75.6091, -75.6047, -75.6003, -75.5961, -75.5920,
    -75.5880, -75.5842, -75.5805, -75.5769, -75.5735, -75.5703, -75.5672, -75.5642,
])

assert len(ref_fci_sa) == len(dists)

ener_dmrg = []
ener_pdmrg = []
ener_uc_nevpt2 = []
ener_uc_mrrept2 = []
ener_uc_mrcisd = []
ener_sc_nevpt2 = []
ener_ic_nevpt2 = []
ener_ic_mrrept2 = []
ener_ic_mrcisd = []
ener_ic_mrcisddq = []

with open("../08-C2/00-dmrg.out") as f:
    for l in f.readlines():
        if l.startswith('DMRG (M=3000) energy ='):
            ener_dmrg.append(float(l.split()[4]))

with open("../08-C2/04-pdmrg.out") as f:
    for l in f.readlines():
        if l.startswith('E(PDMRG) ='):
            ener_pdmrg.append(float(l.split()[2]))

with open("../08-C2/01-uc-ic-1.out") as f:
    for l in f.readlines():
        if l.startswith('E(MRCI) ='):
            ener_uc_mrcisd.append(float(l.split()[2]))

with open("../08-C2/01-uc-ic-2.out") as f:
    for l in f.readlines():
        if l.startswith('E(MRREPT2) ='):
            ener_uc_mrrept2.append(float(l.split()[2]))

with open("../08-C2/01-uc-ic-3.out") as f:
    for l in f.readlines():
        if l.startswith('E(NEVPT2) ='):
            ener_uc_nevpt2.append(float(l.split()[2]))

with open("../08-C2/01-uc-ic-4.out") as f:
    for l in f.readlines():
        if l.startswith('E(WickICMRCISD)   ='):
            ener_ic_mrcisd.append(float(l.split()[2]))
        elif l.startswith('E(WickICMRCISD+Q) ='):
            ener_ic_mrcisddq.append(float(l.split()[2]))

with open("../08-C2/01-uc-ic-7.out") as f:
    for l in f.readlines():
        if l.startswith('E(WickSCNEVPT2) ='):
            ener_sc_nevpt2.append(float(l.split()[2]))

with open("../08-C2/01-uc-ic-6.out") as f:
    for l in f.readlines():
        if l.startswith('E(WickICNEVPT2) ='):
            ener_ic_nevpt2.append(float(l.split()[2]))

with open("../08-C2/01-uc-ic-5.out") as f:
    for l in f.readlines():
        if l.startswith('E(WickICMRREPT2) ='):
            ener_ic_mrrept2.append(float(l.split()[2]))


ener_dmrg = np.array(ener_dmrg)
ener_pdmrg = np.array(ener_pdmrg)
assert len(ener_pdmrg) == len(dists)

# https://coolors.co/a8cbf0-4e4187-eed6bf-cbe1ce-8eb8a7
# xcolors = ["#EED6BF", "#CBE1CE", "#7A7A7A", "#A8CBF0"]
mcolors = ["#ACACAC", "#818181", "#5C5C5C", "#2E2E2E"]
marker = ['o', '^', 's', 'p']

axs[0].grid(which='major', axis='both', alpha=0.5)
axs[0].plot(dists, abs(ener_pdmrg - ener_dmrg), linestyle='dotted', marker=marker[0], mfc='white', mec=mcolors[0],
    linewidth=1.5, markersize=5, color=mcolors[0], label='Perturbative DMRG')
axs[0].plot(dists, abs(ener_uc_nevpt2 - ener_dmrg), '--', marker=marker[1], mfc='white', mec=mcolors[1],
    linewidth=1.5, markersize=5, color=mcolors[1], label='Uncontracted NEVPT2')
axs[0].plot(dists, abs(ener_uc_mrrept2 - ener_dmrg), '-.', marker=marker[2], mfc='white', mec=mcolors[2],
    linewidth=1.5, markersize=4, color=mcolors[2], label='Uncontracted MRREPT2')
axs[0].plot(dists, abs(ener_uc_mrcisd - ener_dmrg), '-', marker=marker[3], mfc='white', mec=mcolors[3],
    linewidth=1.5, markersize=5, color=mcolors[3], label='Uncontracted MRCISD')
axs[0].set_xlim(1.80, 4.15)
# plt.ylim(0, 9)
# plt.xscale('log')
axs[0].set_yscale('log')
# plt.xticks(bdims, list(map(str, bdims)))


axs[1].grid(which='major', axis='both', alpha=0.5)
axs[1].plot(dists, abs(ener_sc_nevpt2 - ener_dmrg), linestyle='dotted', marker=marker[0], mfc='white', mec=mcolors[0],
    linewidth=1.5, markersize=5, color=mcolors[0], label='Strongly contracted NEVPT2')
axs[1].plot(dists, abs(ener_ic_nevpt2 - ener_dmrg), '--', marker=marker[1], mfc='white', mec=mcolors[1],
    linewidth=1.5, markersize=5, color=mcolors[1], label='Internally contracted NEVPT2')
axs[1].plot(dists, abs(ener_ic_mrrept2 - ener_dmrg), '-.', marker=marker[2], mfc='white', mec=mcolors[2],
    linewidth=1.5, markersize=4, color=mcolors[2], label='Internally contracted MRREPT2')
axs[1].plot(dists, abs(ener_ic_mrcisd - ener_dmrg), '-', marker=marker[3], mfc='white', mec=mcolors[3],
    linewidth=1.5, markersize=5, color=mcolors[3], label='Fully internally contracted MRCISD')
axs[1].set_xlim(1.80, 4.15)
axs[1].set_yscale('log')

axs[0].set_ylim(5E-6, 7E-2)
axs[1].set_ylim(5E-6, 7E-2)
axs[0].set_ylabel("Energy error $|E - E_{\\mathrm{DMRG}}|$ (Hartree)")
axs[1].set_ylabel("Energy error $|E - E_{\\mathrm{DMRG}}|$ (Hartree)")
axs[1].set_xlabel("C$-$C bond length ($\\mathrm{\\AA}$)")

axs[0].text(1.85, 3.5E-2, '(a)', color='#000000', fontsize=14)
axs[1].text(1.85, 3.5E-2, '(b)', color='#000000', fontsize=14)

axs[1].add_artist(AnnotationBbox(imagebox, (2.15, 1.35E-5), frameon=False))
axs[1].text(1.93, 1.15E-5, '$\mathrm{C}$', color='#000000', fontsize=16, zorder=999)
axs[1].text(2.29, 1.15E-5, '$\mathrm{C}$', color='#000000', fontsize=16, zorder=999)
axs[0].legend(loc='lower right', fontsize=10)
axs[1].legend(loc='lower right', fontsize=10)
plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.08, wspace=0.0, hspace=0.0)
plt.savefig('fig-08.pdf', dpi=600)
