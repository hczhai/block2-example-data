
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import numpy as np

driver = DMRGDriver(scratch="/central/scratch/hczhai/st-00", symm_type=SymmetryTypes.SU2, stack_mem=100 << 30, n_threads=24, mpi=True)
driver.read_fcidump('./FCIDUMP-21', pg='c1')
driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

bond_dims = [500] * 4 + [1000] * 4 + [2000] * 4 + [3000] * 4 + [4000] * 4 + [5000] * 4 + [6000] * 4
noises = [1e-4] * 12 + [1e-5] * 16 + [0]
thrds = [1e-5] * 8 + [1E-6] * 8 + [1E-7] * 12 + [1E-8]
n_sweeps = 40

mpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, reorder="fiedler", algo_type=MPOAlgorithmTypes.Conventional, iprint=2)
ket = driver.get_random_mps(tag="KET", bond_dim=500, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds,
    twosite_to_onesite=34, tol=1E-12, cutoff=1E-24, iprint=2)
print('DMRG energy = %20.15f' % energy)
dm1 = driver.get_1pdm(ket)
if driver.mpi.rank == driver.mpi.root:
    np.save('00-1pdm.npy', dm1)

