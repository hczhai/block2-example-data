
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

def get_ueg_integrals(l, cutoff):
    im = int(np.ceil(l * (2 * cutoff) ** 0.5 / (2 * np.pi)))
    basis = np.mgrid[-im:im, -im:im, -im:im]
    basis = basis[:, 4 * np.pi ** 2 / l ** 2 * np.sum(basis ** 2, axis=0) < 2 * cutoff].T
    basis = 2 * np.pi * basis[np.argsort(np.sum(basis ** 2, axis=1), kind='stable')] / l
    h1e = np.diag(np.sum(basis ** 2, axis=1) / 2)
    q = basis[:, None] - basis[None, :]
    qsq = np.sum(q ** 2, axis=2)
    g2e = 4 * np.pi / l ** 3 * (qsq != 0) / (qsq + (qsq == 0))
    g2e = g2e[:, :, None, None] * (np.sum((q[:, :, None, None] + q[None, None]) ** 2, axis=4) == 0)
    return h1e, g2e

rs = 4.0
ne = 4
l = (4 * np.pi / 3 * ne) ** (1 / 3) * rs
cutoff = 2.0 * np.pi ** ne / (l * l * 3) + 1E-5
h1e, g2e = get_ueg_integrals(l, cutoff)

driver = DMRGDriver(scratch="/scratch/global/hczhai/UEG", symm_type=SymmetryTypes.SU2, stack_mem=100 << 30, n_threads=24, mpi=False)
driver.initialize_system(n_sites=len(h1e), n_elec=ne, spin=0, orb_sym=None)

bond_dims = [250] * 4 + [500] * 4 + [1200] * 4
noises = [1e-4] * 4 + [1e-5] * 8 + [0]
thrds = [1e-6] * 4 + [1e-10] * 8

n_sweeps = 20
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=0.0, iprint=2)
ket = driver.get_random_mps(tag="KET", bond_dim=500, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds,
    twosite_to_onesite=12, tol=1E-12, cutoff=1E-24, iprint=2)
print('Ground-State DMRG energy = %20.15f' % energy)

mu = -2
n_h1e = np.identity(len(h1e))
ft_mpo = driver.get_qc_mpo(h1e=h1e - mu * n_h1e, g2e=g2e, ecore=0.0, iprint=2, ancilla=True)
print(ft_mpo.n_sites)
ft_mps = driver.get_ancilla_mps(tag="FKET")
print(ft_mps.n_sites)

b = driver.expr_builder()
b.add_term('(C+D)0', [i for i in range(len(h1e)) for _ in [0, 1]], np.sqrt(2))
n_mpo = driver.get_mpo(b.finalize(), iprint=1, ancilla=True)

b = driver.expr_builder()
b.add_term('((C+D)0+(C+D)0)0', [x for i in range(len(h1e)) for j in range(len(h1e)) for x in [i, i, j, j]], 2)
nn_mpo = driver.get_mpo(b.finalize(), iprint=1, ancilla=True)

e_mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=0.0, iprint=2, ancilla=True)
i_mpo = driver.get_identity_mpo(ancilla=True)

beta = 10.0 # target inverse temperature
dbeta = 0.025 # imag time step
n_steps = int(beta / dbeta + 1E-8)
bdims = [1200]

dt = dbeta / 2
eners = []
for i in range(n_steps):
    cur_beta = i * dbeta
    i_expt = driver.expectation(ft_mps, i_mpo, ft_mps)
    e_expt = driver.expectation(ft_mps, e_mpo, ft_mps) / i_expt
    n_expt = driver.expectation(ft_mps, n_mpo, ft_mps) / i_expt
    nn_expt = driver.expectation(ft_mps, nn_mpo, ft_mps) / i_expt
    print('Beta = %7.3f Norm = %15.8f <E> = %15.8f <N> = %15.8f <N^2> = %15.8f' % (
        cur_beta, i_expt, e_expt, n_expt, nn_expt))
    eners.append(e_expt)
    ft_mps = driver.td_dmrg(ft_mpo, ft_mps, dt, n_steps=1, final_mps_tag="FKET",
        n_sub_sweeps=8 if i == 0 else 2, bond_dims=bdims, normalize_mps=True, iprint=2)

