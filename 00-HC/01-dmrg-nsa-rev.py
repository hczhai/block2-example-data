
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import numpy as np

driver = DMRGDriver(scratch="/central/scratch/hczhai/st-00-nsa", symm_type=SymmetryTypes.SZ, stack_mem=100 << 30, n_threads=24, mpi=True)
driver.read_fcidump('./FCIDUMP-21', pg='c1')
driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

bond_dims = [5000] * 4 + [4000] * 4 + [3000] * 4 + [2000] * 4 + [1000] * 4
noises = [0] * 20
thrds = [1e-8] * 20
n_sweeps = 20

ket = driver.load_mps(tag='KET').deep_copy('KET-TMP')
ket = driver.adjust_mps(ket, dot=2)[0]
mpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, reorder="fiedler", algo_type=MPOAlgorithmTypes.Conventional, iprint=2)
energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds,
    dav_max_iter=4, tol=0, cutoff=1E-24, iprint=2)
