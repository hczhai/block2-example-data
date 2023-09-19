
import numpy as np, time
from pyscf import gto, scf, mcscf, dmrgscf, lib
import os, sys

ith = int(sys.argv[1])

theory = [
    "dmrgscf",
    "uc-mrcisd",
    "uc-mrrept2",
    "uc-nevpt2",
    "ic-mrcisd",
    "ic-mrrept2",
    "ic-nevpt2",
    "sc-nevpt2",
][ith]

if theory == "uc-mrcisd":
    from pyblock2.mrpt import UCMRCISD as MR
elif theory == "uc-mrrept2":
    from pyblock2.mrpt import UCMRREPT2 as MR
elif theory == "uc-nevpt2":
    from pyblock2.mrpt import UCNEVPT2 as MR
elif theory == "ic-mrcisd":
    from pyblock2.icmr.icmrcisd_full import ICMRCISD as MR
elif theory == "ic-mrrept2":
    from pyblock2.icmr.icmrrept2_full import ICMRREPT2 as MR
elif theory == "ic-nevpt2":
    from pyblock2.icmr.icnevpt2_full import ICNEVPT2 as MR
elif theory == "sc-nevpt2":
    from pyblock2.icmr.scnevpt2 import SCNEVPT2 as MR
elif theory != "dmrgscf":
    raise RuntimeError('unsupported theory' + theory)

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

lib.param.TMPDIR = '/central/scratch/hczhai/st-07-01-%s' % theory

if not os.path.exists(lib.param.TMPDIR):
    os.mkdir(lib.param.TMPDIR)

mcscf.casci.FRAC_OCC_THRESHOLD = -1

dists = np.arange(1.80, 4.16, 0.05)
eners = np.zeros_like(dists)
eners_dq = np.zeros_like(dists)

cas_cur = []
ic_cur = []
for ix, d in enumerate(dists):

    tstart = time.perf_counter()

    mol = gto.M(atom='C 0 0 0; C 0 0 %f' % d, unit='bohr',
        symmetry='d2h', basis='cc-pvdz', spin=0, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)
    mc = mcscf.CASSCF(mf, 8, 8)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=1000, tol=1e-10)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = 24
    mc.conv_tol = 1e-10
    mc.fcisolver.conv_tol = 1e-12
    mc.canonicalization = True
    mc.natorb = True
    mc.mc2step()

    mol.verbose = 5
    mol.symmetry = 'c1'
    mol.build()

    if "uc" in theory:
        mr = MR(mc).set(scratch=lib.param.TMPDIR).run()
        eners[ix] = mr.e_tot
    elif "ic-mrcisd" in theory:
        mr = MR(mc).run()
        eners[ix] = mr.e_tot
        eners_dq[ix] = mr.e_tot + mr.de_dav_q
    elif "dmrgscf" in theory:
        eners[ix] = mc.e_tot
    else:
        mr = MR(mc).run()
        eners[ix] = mr.e_tot

    print('Time cost = %10.3f' % (time.perf_counter() - tstart))

np.save('%s.npy' % theory, np.array(eners))
if "ic-mrcisd" in theory:
    np.save('%s-dq.npy' % theory, np.array(eners_dq))
