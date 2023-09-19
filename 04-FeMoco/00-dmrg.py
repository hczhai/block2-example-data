
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import numpy as np

driver = DMRGDriver(scratch="/central/scratch/hczhai/st-04", symm_type=SymmetryTypes.SU2, stack_mem=100 << 30, n_threads=24, mpi=False)
driver.read_fcidump('./FCIDUMP', pg='c1')
driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)
driver.h1e[np.abs(driver.h1e) < 1e-12] = 0
driver.g2e[np.abs(driver.g2e) < 1e-12] = 0

mpo = driver.get_qc_mpo(driver.h1e, driver.g2e, ecore=driver.const_e, iprint=2,
    algo_type=MPOAlgorithmTypes.FastBipartite, cutoff=1E-12, integral_cutoff=1E-12)

mpo = driver.get_qc_mpo(driver.h1e, driver.g2e, ecore=driver.const_e, iprint=2,
    algo_type=MPOAlgorithmTypes.Conventional, cutoff=1E-12, integral_cutoff=1E-12)

for ct in [1E-5, 1E-6, 1E-7, 1E-8, 1E-9, 1E-10, 1E-11, 1E-12]:
    mpo = driver.get_qc_mpo(driver.h1e, driver.g2e, ecore=driver.const_e, iprint=2,
        algo_type=MPOAlgorithmTypes.FastBlockedSVD, cutoff=ct, integral_cutoff=1E-12)
