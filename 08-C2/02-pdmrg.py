
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
    driver = DMRGDriver(scratch='/central/scratch/hczhai/st-07-02', symm_type=SymmetryTypes.SU2, stack_mem=70 << 30, n_threads=24)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1, reorder='irrep')
    mpo_en = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=0.0, iprint=1, reorder='irrep', esptein_nesbet_partition=True)

    bond_dims = [150] * 16
    noises = [1e-4] * 4 + [1e-5] * 8 + [0]
    thrds = [1e-10] * 16
    ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
    e0 = driver.dmrg(mpo, ket, n_sweeps=20, twosite_to_onesite=12,
        bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=2)

    print('DMRG (M=150) energy = %20.15f' % e0)

    ratio = 0.5
    ex = driver.expectation(ket, mpo_en, ket, iprint=2)
    mpo_en.const_e = mpo_en.const_e - ratio * (e0 - ecore) - (1 - ratio) * ex
    mpo.const_e = mpo.const_e - e0

    bond_dims = [250] * 8 + [500] * 8 + [1000] * 8 + [1800] * 8
    noises = [1e-4] * 24 + [5e-5] * 8 + [0]
    thrds = [2E-6] * 24 + [1E-6] * 8 + [2E-11] * 8

    ket = driver.adjust_mps(ket, dot=2)[0]
    bra = driver.get_random_mps(tag="BRA", bond_dim=250, nroots=1, center=ket.center)
    xx = driver.multiply(bra, mpo, ket, n_sweeps=40, bra_bond_dims=bond_dims,
        noises=noises, thrds=thrds, left_mpo=mpo_en, linear_max_iter=400, iprint=1)

    print('E(PDMRG) = %20.15f' % (e0 - xx))
    print('Time cost = %10.3f' % (time.perf_counter() - tstart))

    eners[ix] = e0 - xx

np.save('pdmrg.npy', np.array(eners))
