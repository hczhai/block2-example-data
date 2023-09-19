
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

# Physical Review B 79, 235336 (2009) Eqs. (1) and (2)
def siam(n, tp, t, vg, u, v=0.0):
    assert n % 2 == 0
    idximp = n // 2 - 1
    idxls = np.arange(0, idximp, dtype=int)
    idxrs = np.arange(idximp + 1, n, dtype=int)
    h1e = np.zeros((n, n))
    g2e = np.zeros((n, n, n, n))
    g2e[idximp, idximp, idximp, idximp] = u / 2
    h1e[idximp, idximp] = vg
    h1e[idximp, idxls[-1]] = h1e[idxls[-1], idximp] = -tp
    h1e[idximp, idxrs[0]] = h1e[idxrs[0], idximp] = -tp
    for il, ilp in zip(idxls, idxls[1:]):
        h1e[ilp, il] = h1e[il, ilp] = -t
    for ir, irp in zip(idxrs, idxrs[1:]):
        h1e[irp, ir] = h1e[ir, irp] = -t
    for il in idxls:
        h1e[il, il] = -v / 2
    for ir in idxrs:
        h1e[ir, ir] = v / 2
    return h1e, g2e, idximp

n, tp, vpu, u = 48, 0.3535, 2.0, 0.5
gs_siam = siam(n=n, tp=tp, t=1, vg=-u / 2, u=u)
td_siam = siam(n=n, tp=tp, t=1, vg=-u / 2, u=u, v=vpu * u)

driver = DMRGDriver(scratch="/scratch/global/hczhai/nodex5x",
                    symm_type=SymmetryTypes.SU2 | SymmetryTypes.CPX, stack_mem=50 << 30, n_threads=28, mpi=False)
driver.initialize_system(n_sites=n, n_elec=n)

h1e, g2e, idximp = gs_siam

# particle number operator
b = driver.expr_builder()
b.add_term('(C+D)0', [i for i in range(n) for _ in [0, 1]], np.sqrt(2))
nmpo = driver.get_mpo(b.finalize(), iprint=1)

# current operator
f = -0.5 * np.sqrt(2)
b = driver.expr_builder()
b.add_term('(C+D)0', [idximp, idximp - 1, idximp - 1, idximp, idximp + 1, idximp, idximp, idximp + 1], [f, -f, f, -f])
jmpo = driver.get_mpo(b.finalize(), iprint=1)

impo = driver.get_identity_mpo()

# ground state

bond_dims = [250] * 4 + [500] * 4 + [1200] * 4
noises = [1e-4] * 4 + [1e-5] * 8 + [0]
thrds = [1e-6] * 4 + [1e-10] * 8

mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=0.0, iprint=1)
ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=2)
print('DMRG energy = %20.15f' % energy)

# real-time evolution

tmax = 30
n_steps = 600
dt = tmax / n_steps
td_bdim = [1000]

h1e, g2e, idximp = td_siam
td_mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=0.0, iprint=1)

j_arr = np.zeros((n_steps, ))

i_expt = driver.expectation(ket, impo, ket)
j_expt = driver.expectation(ket, jmpo, ket) / i_expt
j_expt *= 1j * tp * 2 * np.pi / (vpu * u)
j_arr[0] = j_expt.real

for it in range(n_steps - 1):
    cur_t = (it + 1) * dt
    ket = driver.td_dmrg(td_mpo, ket, target_t=dt * 1j, te_type='ts', n_steps=1, n_sub_sweeps=1,
        normalize_mps=False, final_mps_tag='KET', bond_dims=td_bdim, iprint=2)
    i_expt = driver.expectation(ket, impo, ket)
    j_expt = driver.expectation(ket, jmpo, ket) / i_expt
    n_expt = driver.expectation(ket, nmpo, ket) / i_expt
    e_expt = driver._te.energies[-1]
    j_expt *= 1j * tp * 2 * np.pi / (vpu * u)
    print('T = %7.3f Norm = %15.8f + %15.8f i <E> = %15.8f + %15.8f i <N> = %15.8f + %15.8f i <J> = %15.8f + %15.8f i' % (
        cur_t, i_expt.real, i_expt.imag, e_expt.real, e_expt.imag, n_expt.real, n_expt.imag, j_expt.real, j_expt.imag))
    j_arr[it + 1] = j_expt.real

