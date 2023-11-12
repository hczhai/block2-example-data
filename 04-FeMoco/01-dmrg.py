
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import time
import numpy as np

driver = DMRGDriver(scratch="/central/scratch/hczhai/st-04", symm_type=SymmetryTypes.SU2, stack_mem=240 << 30, n_threads=64, mpi=True, restart_dir='./MPS-2000')
driver.read_fcidump('./FCIDUMP', pg='c1')
driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)
driver.h1e[np.abs(driver.h1e) < 1e-12] = 0
driver.g2e[np.abs(driver.g2e) < 1e-12] = 0

driver.mpi.barrier()
tt = time.perf_counter()
driver.mpi.barrier()

mpo = driver.get_qc_mpo(driver.h1e, driver.g2e, ecore=driver.const_e, iprint=2,
   algo_type=MPOAlgorithmTypes.Conventional, cutoff=1E-12, integral_cutoff=1E-12)

driver.mpi.barrier()
print('ex time = ', time.perf_counter() - tt)
driver.mpi.barrier()

ket = driver.get_random_mps(tag="KET", bond_dim=500, nroots=1)
bond_dims =  [500] * 5 + [1000] * 5 + [1500] * 5 + [2000] * 20
noises    = [1e-3] * 5 + [1e-4] * 5 + [1e-5] * 12 + [0]
thrds     = [1e-4] * 5 + [1e-5] * 5 + [5e-6] * 25

driver.mpi.barrier()
tt = time.perf_counter()
driver.mpi.barrier()

energies = driver.dmrg(mpo, ket, n_sweeps=35, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=2, dav_max_iter=400, cutoff=1E-24, twosite_to_onesite=32)

driver.mpi.barrier()
print('ex time = ', time.perf_counter() - tt)
driver.mpi.barrier()

print(energies)
