
import time
from datetime import datetime
txst = time.perf_counter()
print("START  TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
spin = 10

from pyscf import scf, lib, symm
import numpy as np
mfchk = "../mf-1/mf.chk"
mol, mfx = scf.chkfile.load_scf(mfchk)
if spin is not None:
    mol.spin = spin
    mol.build()
mf = scf.sfx2c(scf.UHF(mol))
mf.chkfile = "mf.chk"
mf.mo_coeff = mfx["mo_coeff"]
mf.mo_energy = mfx["mo_energy"]
mf.mo_occ = mfx["mo_occ"]

ncore, ncas = 26, 30
xcc_nelec = None
xcc_ncas = None

txx = time.perf_counter()
do_ccsd_t = True
do_st_extrap = False
mf.max_memory = 100000
do_st_extrap = True

from pyscf import cc
nfrozen = 41
if xcc_ncas is not None:
    xna = (xcc_nelec + mol.spin) // 2
    xnb = (xcc_nelec - mol.spin) // 2
    mc = XUCCSD(mf, xcc_ncas, (xna, xnb), frozen=nfrozen)
else:
    mc = cc.UCCSD(mf, frozen=nfrozen)
mc.max_cycle = 1000
eris = mc.ao2mo(mc.mo_coeff)
mc.kernel(eris=eris)
tt1, tt2 = mc.t1, mc.t2
print('ECCSD    = ', mc.e_tot)

if do_ccsd_t:
    e_t_all = mc.ccsd_t(eris=eris)
    print("ECCSD(T) = ", e_t_all + mc.e_tot)

    # from pyblock2.cc.uccsd import wick_t3_amps, wick_ccsd_t
    # t3 = wick_t3_amps(mc, mc.t1, mc.t2, eris=eris)
    # for t3x in t3:
    #     ged = t3x.shape[:3]
    #     xst, xed = ncore, ncore + ncas
    #     t3x[xst:, xst:, xst:, :xed - ged[0], :xed - ged[1], :xed - ged[2]] = 0
    # e_t_no_cas = wick_ccsd_t(mc, mc.t1, mc.t2, eris=eris, t3=t3)

    # require a custom version of pyscf
    # https://github.com/hczhai/pyscf/tree/ccsd_t_cas
    from pyscf.cc.uccsd_t import _gen_contract_aaa
    import inspect
    assert "cas_exclude" in inspect.getfullargspec(_gen_contract_aaa).args

    eris.cas_exclude = ncore, mc.t1[0].shape[-1] - (ncore + ncas - mc.t1[0].shape[0])
    print('\ncas_exclude = ', eris.cas_exclude)
    e_t_no_cas = mc.ccsd_t(eris=eris)

    print("E(T) = ", e_t_no_cas, '\n')
del mc
del eris

from pyblock2._pyscf.ao2mo import get_uhf_integrals
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
from pyblock2.driver.core import STTypes, SimilarityTransform
import numpy as np
import os

def fmt_size(i, suffix='B'):
    if i < 1000:
        return "%d %s" % (i, suffix)
    else:
        a = 1024
        for pf in "KMGTPEZY":
            p = 2
            for k in [10, 100, 1000]:
                if i < k * a:
                    return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                p -= 1
            a *= 1024
    return "??? " + suffix

_, n_elec, spin, ecore, h1e, g2e, orb_sym = get_uhf_integrals(mf, ncore=41)
e_hf = mf.e_tot
del mf

try:
    import psutil
    mem = psutil.Process(os.getpid()).memory_info().rss
    print("pre-st memory usage = ", fmt_size(mem))
except ImportError:
    pass

dtotal = 0
for k, dx in [('t1', tt1), ('t2', tt2), ('h1e', h1e), ('g2e', g2e)]:
    for ddx in dx:
        print(k, "data memory = ", fmt_size(ddx.nbytes))
        dtotal += ddx.nbytes
print("total data memory = ", fmt_size(dtotal))

print('ecore = ', ecore)
print('orb_sym = ', orb_sym)

scratch = lib.param.TMPDIR

print("PART TIME (PRE)  = %20.3f" % (time.perf_counter() - txx))
txx = time.perf_counter()

driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SZ,
                    stack_mem=int(mol.max_memory * 1000 ** 2), n_threads=int(os.environ["OMP_NUM_THREADS"]))
driver.integral_symmetrize(orb_sym, h1e=h1e, g2e=g2e, iprint=1)
for ttx in tt1 + tt2:
    assert np.array(orb_sym).ndim == 1
    orb_syms = []
    for ip in ttx.shape[:ttx.ndim // 2]:
        orb_syms.append(orb_sym[:ip])
    for ip in ttx.shape[ttx.ndim // 2:]:
        orb_syms.append(orb_sym[-ip:])
    driver.integral_symmetrize(orb_syms, hxe=ttx, iprint=1)
dt, ecore, ncas, n_elec = SimilarityTransform.make_sz(h1e, g2e, ecore, tt1, tt2, scratch,
    n_elec, ncore=ncore, ncas=ncas, st_type=STTypes.H_HT_HTT, iprint=2)
del h1e, g2e, tt1, tt2

try:
    import psutil
    mem = psutil.Process(os.getpid()).memory_info().rss
    print("pre-dmrg memory usage = ", fmt_size(mem))
except ImportError:
    pass

print("PART TIME (ST)  = %20.3f" % (time.perf_counter() - txx))
txx = time.perf_counter()

print('neleccas =', n_elec, 'ncas =', ncas, 'spin = ', spin)

driver.initialize_system(
    n_sites=ncas, n_elec=n_elec, spin=spin,
    orb_sym=np.array(orb_sym)[..., ncore:ncore + ncas], pg_irrep=0
)

b = driver.expr_builder()
for expr, v in dt.items():
    print('expr = ', expr)
    b.add_sum_term(expr, np.load(v), cutoff=1E-13)
b.add_const(ecore)
print('ok')

for k in os.listdir(scratch):
    if k.endswith(".npy") and k.startswith("ST-DMRG."):
        os.remove(scratch + "/" + k)

print("PART TIME (TERM)  = %20.3f" % (time.perf_counter() - txx))
txx = time.perf_counter()

mpo = driver.get_mpo(b.finalize(), algo_type=MPOAlgorithmTypes.FastBipartite, iprint=2)

print("PART TIME (MPO)  = %20.3f" % (time.perf_counter() - txx))
txx = time.perf_counter()

ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
bond_dims = [400] * 4 + [800] * 8
noises = [1e-5] * 4 + [1e-6] * 4 + [0]
thrds = [1e-7] * 4 + [1e-9] * 4
e_st = driver.dmrg(
    mpo,
    ket,
    n_sweeps=20,
    dav_type="NonHermitian",
    bond_dims=bond_dims,
    noises=noises,
    thrds=thrds,
    iprint=2,
)

print('EST    = ', e_st)
if e_t_no_cas is not None:
    print('EST(T) = ', e_st + e_t_no_cas)
print("PART TIME (DMRG)  = %20.3f" % (time.perf_counter() - txx))

if do_st_extrap:
    ket = ket.deep_copy('GS-TMP')
    bond_dims = [800] * 4 + [700] * 4 + [600] * 4 + [500] * 4 + [400] * 4 + [300] * 4 + [200] * 4 + [100] * 4
    noises = [0] * 32
    thrds = [1e-12] * 32
    energy = driver.dmrg(mpo, ket, n_sweeps=32, bond_dims=bond_dims, noises=noises,
        dav_type="NonHermitian", tol=0, thrds=thrds, iprint=2)

for k in os.listdir(scratch):
    if '.PART.' in k:
        os.remove(scratch + "/" + k)


txed = time.perf_counter()
print("FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("TOTAL TIME  = %20.3f" % (txed - txst))
