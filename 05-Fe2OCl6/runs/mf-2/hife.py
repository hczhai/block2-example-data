
import time
from datetime import datetime
txst = time.perf_counter()
print("START  TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyscf import gto, scf
mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
Fe     1.677856070000   0.000522330000   0.064759320000
Fe    -1.677856070000  -0.000522330000   0.064759320000
O      0.000000000000   0.000000000000  -0.470990740000
Cl     1.870027040000  -1.097964370000   1.990916820000
Cl     2.932449170000  -0.982104880000  -1.474672880000
Cl     2.371609360000   2.079540910000  -0.504465910000
Cl    -1.870027040000   1.097964370000   1.990916820000
Cl    -2.932449170000   0.982104880000  -1.474672880000
Cl    -2.371609360000  -2.079540910000  -0.504465910000

'''
mol.basis = "ccpvdz-dk"
mol.spin = 0
mol.charge = -2
mol.max_memory = 82000

mol.build()
print("NAO   = ", mol.nao)
print("NELEC = ", mol.nelec)
dm = None

from pyblock2._pyscf import scf as b2scf
dm = b2scf.get_metal_init_guess(mol, orb="Fe 3d", atom_idxs=[0, 1], coupling="+-", atomic_spin=5)


print("PG = ", mol.groupname)

mf = scf.sfx2c(scf.UHF(mol))
mf.chkfile = 'mf.chk'
mf.conv_tol = 1E-12
mf.max_cycle = 1000
mf = mf.newton()

mf.kernel(dm0=dm)
dm = mf.make_rdm1()

import numpy as np
np.save("mf_occ.npy", mf.mo_occ)
np.save("mo_coeff.npy", mf.mo_coeff)
np.save("mo_energy.npy", mf.mo_energy)
np.save("e_tot.npy", mf.e_tot)
np.save("mf_dmao.npy", dm)

from pyblock2._pyscf import scf as b2scf
b2scf.mulliken_pop_dmao(mol, dm)


txed = time.perf_counter()
print("FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("TOTAL TIME  = %20.3f" % (txed - txst))
