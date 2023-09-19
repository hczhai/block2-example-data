
import numpy as np, time
from pyscf import gto, scf
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

dists = np.arange(1.80, 4.16, 0.05)
eners = np.zeros_like(dists)

for ix, d in enumerate(dists):

    tstart = time.perf_counter()

    mol = gto.M(atom='C 0 0 0; C 0 0 %f' % d, unit='bohr', symmetry='d2h', basis='cc-pvdz', spin=0, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, g2e_symm=1)
    driver = DMRGDriver(scratch='/central/scratch/hczhai/st-07-00', symm_type=SymmetryTypes.SU2, stack_mem=70 << 30, n_threads=24)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1, reorder='irrep')

    bond_dims = [250] * 8 + [500] * 8 + [1000] * 8 + [1800] * 8 + [3000] * 8
    noises = [1e-4] * 24 + [5e-5] * 16 + [0]
    thrds = [1E-5] * 24 + [5E-6] * 8 + [1E-9] * 16

    ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
    e0 = driver.dmrg(mpo, ket, n_sweeps=48, twosite_to_onesite=42,
        bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=2)

    print('DMRG (M=3000) energy = %20.15f' % e0)
    print('Time cost = %10.3f' % (time.perf_counter() - tstart))

    eners[ix] = e0

np.save('dmrg.npy', np.array(eners))
