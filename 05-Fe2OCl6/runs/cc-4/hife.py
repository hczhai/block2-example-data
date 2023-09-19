
import time
from datetime import datetime
txst = time.perf_counter()
print("START  TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
spin = 0

from pyscf import scf, lib, symm
import numpy as np
mfchk = "../mf-2/mf.chk"
mol, mfx = scf.chkfile.load_scf(mfchk)
if spin is not None:
    mol.spin = spin
    mol.build()
mf = scf.sfx2c(scf.UHF(mol))
mf.chkfile = "mf.chk"
mf.mo_coeff = mfx["mo_coeff"]
mf.mo_energy = mfx["mo_energy"]
mf.mo_occ = mfx["mo_occ"]

do_ccsd_t = True
bcc = False
do_spin_square = False
nat_with_pg = False
save_amps = False
xcc_nelec = None
xcc_ncas = None
no_cc = False

from sys import argv
import os
is_restart = len(argv) >= 2 and argv[1] == "1"

if not is_restart:
    for fname in ['/ccdiis.h5', '/ccdiis-lambda.h5']:
        if os.path.isfile(lib.param.TMPDIR + fname):
            fid = 1
            while os.path.isfile(lib.param.TMPDIR + fname + '.%d' % fid):
                fid += 1
            os.rename(lib.param.TMPDIR + fname,
                lib.param.TMPDIR + fname + '.%d' % fid)

print('mf occ', np.sum(mf.mo_occ, axis=-1), mf.mo_occ)

print('ref energy', mf.energy_tot())
if no_cc:
    quit()

from pyscf import cc, scf

nfrozen = 41
if xcc_ncas is not None:
    if isinstance(mf, scf.uhf.UHF):
        xna = (xcc_nelec + spin) // 2
        xnb = (xcc_nelec - spin) // 2
        mc = XUCCSD(mf, xcc_ncas, (xna, xnb), frozen=nfrozen)
    else:
        mc = XRCCSD(mf, xcc_ncas, xcc_nelec, frozen=nfrozen)
else:
    mc = cc.CCSD(mf, frozen=nfrozen)
mc.diis_file = lib.param.TMPDIR + '/ccdiis.h5'
mc.max_cycle = 1000
cc.incore_complete = True

if is_restart and os.path.isfile(lib.param.TMPDIR + '/ccdiis.h5'):
    print("restart ccsd from ", lib.param.TMPDIR + '/ccdiis.h5')
    mc.restore_from_diis_(lib.param.TMPDIR + '/ccdiis.h5')
    t1, t2 = mc.t1, mc.t2
    mc.kernel(t1, t2)
else:
    mc.kernel()
e_ccsd = mc.e_tot
print('ECCSD    = ', e_ccsd)
print("PART TIME (CCSD) = %20.3f" % (time.perf_counter() - txst))

if save_amps:
    np.save("ccsd_t1.npy", mc.t1)
    np.save("ccsd_t2.npy", mc.t2)

if do_spin_square:
    S2 = mc.spin_square()[0]
    print('CCSD <S^2> = ', S2)
    print("PART TIME (CCSD S2) = %20.3f" % (time.perf_counter() - txst))

if bcc:
    from libdmet.solver.cc import bcc_loop
    mc = bcc_loop(mc, utol=bcc_conv_tol, max_cycle=bcc_max_cycle, verbose=mol.verbose)
    e_bccsd = mc.e_tot
    print('EBCCSD   = ', e_bccsd)
    print("PART TIME (BCCSD) = %20.3f" % (time.perf_counter() - txst))

    if do_spin_square:
        S2 = mc.spin_square()[0]
        print('BCCSD <S^2> = ', S2)
        print("PART TIME (BCCSD S2) = %20.3f" % (time.perf_counter() - txst))

if do_ccsd_t:
    eris = mc.ao2mo()
    e_ccsd_t = mc.e_tot + mc.ccsd_t(eris=eris)
    print('ECCSD(T) = ', e_ccsd_t)
    print("PART TIME (CCSD(T))  = %20.3f" % (time.perf_counter() - txst))

    if do_spin_square:
        from pyscf.cc import uccsd_t_lambda, uccsd_t_rdm
        from pyscf.fci import spin_op
        conv, l1, l2 = uccsd_t_lambda.kernel(mc, tol=1E-7)
        print("PART TIME (CCSD(T) Lambda) = %20.3f" % (time.perf_counter() - txst))
        assert conv
        dm1 = uccsd_t_rdm.make_rdm1(mc, t1, t2, l1, l2, eris)
        print("PART TIME (CCSD(T) RDM1) = %20.3f" % (time.perf_counter() - txst))

        import numpy as np
        if dm1[0].ndim == 2:
            mc_occ_t = np.diag(dm1[0]) + np.diag(dm1[1])
        else:
            mc_occ_t = np.diag(dm1)

        np.save("cc_t_occ.npy", mc_occ_t)
        np.save("cc_t_mo_coeff.npy", mc.mo_coeff)
        np.save("cc_t_e_tot.npy", e_ccsd_t)
        np.save("cc_t_dmmo.npy", dm1)

        nat_occ_t, u_t = np.linalg.eigh(dm1)
        nat_coeff_t = np.einsum('...pi,...ij->...pj', mc.mo_coeff, u_t, optimize=True)
        np.save("cc_t_nat_coeff.npy", nat_coeff_t[..., ::-1])
        np.save("cc_t_nat_occ.npy", nat_coeff_t[..., ::-1])

        print('ccsd(t) nat occ', np.sum(nat_occ_t, axis=-1), nat_occ_t)

        dm2 = uccsd_t_rdm.make_rdm2(mc, t1, t2, l1, l2, eris)
        print("PART TIME (CCSD(T) RDM2) = %20.3f" % (time.perf_counter() - txst))
        S2 = spin_op.spin_square_general(*dm1, *dm2, mc.mo_coeff, mc._scf.get_ovlp())[0]
        print('CCSD(T) <S^2> = ', S2)
        print("PART TIME (CCSD(T) S2) = %20.3f" % (time.perf_counter() - txst))

mc.diis_file = lib.param.TMPDIR + '/ccdiis-lambda.h5'
if is_restart and os.path.isfile(lib.param.TMPDIR + '/ccdiis-lambda.h5'):
    print("restart ccsd-lambda from ", lib.param.TMPDIR + '/ccdiis-lambda.h5')
    from pyscf import lib
    ccvec = lib.diis.restore(lib.param.TMPDIR + '/ccdiis-lambda.h5').extrapolate()
    l1, l2 = mc.vector_to_amplitudes(ccvec)
    mc.restore_from_diis_(lib.param.TMPDIR + '/ccdiis-lambda.h5')
    mc.solve_lambda(mc.t1, mc.t2, l1, l2)
else:
    mc.solve_lambda(mc.t1, mc.t2)
print("PART TIME (CCSD-lambda)  = %20.3f" % (time.perf_counter() - txst))

import numpy as np
dm = mc.make_rdm1()
if dm[0].ndim == 2:
    mc_occ = np.diag(dm[0]) + np.diag(dm[1])
else:
    mc_occ = np.diag(dm)
print("PART TIME (1PDM)  = %20.3f" % (time.perf_counter() - txst))


import numpy as np
np.save("cc_occ.npy", mc_occ)
np.save("cc_mo_coeff.npy", mc.mo_coeff)
np.save("cc_e_tot.npy", mc.e_tot)
np.save("cc_dmmo.npy", dm)

# dmao = np.einsum('xpi,xij,xqj->xpq', mc.mo_coeff, dm, mc.mo_coeff, optimize=True)
# coeff_inv = np.linalg.pinv(mc.mo_coeff)
# dmmo = np.einsum('xip,xpq,xjq->xij', coeff_inv, dmao, coeff_inv, optimize=True)

nat_occ, u = np.linalg.eigh(dm)
nat_coeff = np.einsum('...pi,...ij->...pj', mc.mo_coeff, u, optimize=True)
np.save("nat_coeff.npy", nat_coeff[..., ::-1])
np.save("nat_occ.npy", nat_occ[..., ::-1])

print('nat occ', np.sum(nat_occ, axis=-1), nat_occ)

if nat_with_pg:
    np.save("nat_coeff_no_pg.npy", nat_coeff[..., ::-1])
    np.save("nat_occ_no_pg.npy", nat_occ[..., ::-1])

    orb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mc.mo_coeff, tol=1e-2)
    orb_sym = [symm.irrep_name2id(mol.groupname, ir) for ir in orb_sym]
    if np.array(dm).ndim == 3:
        spdm = np.sum(dm, axis=0)
    else:
        spdm = dm
    n_sites = len(spdm)
    spdm = spdm.flatten()
    nat_occ = np.zeros((n_sites, ))

    import block2 as b
    b.MatrixFunctions.block_eigs(spdm, nat_occ, b.VectorUInt8(orb_sym))
    rot = np.array(spdm.reshape((n_sites, n_sites)).T, copy=True)
    midx = np.argsort(nat_occ)[::-1]
    nat_occ = nat_occ[midx]
    rot = rot[:, midx]
    orb_sym = np.array(orb_sym)[midx]
    for isym in set(orb_sym):
        mask = np.array(orb_sym) == isym
        for j in range(len(nat_occ[mask])):
            mrot = rot[mask, :][:j + 1, :][:, mask][:, :j + 1]
            mrot_det = np.linalg.det(mrot)
            if mrot_det < 0:
                mask0 = np.arange(len(mask), dtype=int)[mask][j]
                rot[:, mask0] = -rot[:, mask0]
    nat_coeff = np.einsum('...pi,...ij->...pj', mc.mo_coeff, rot, optimize=True)
    print('nat occ =', nat_occ)
    print('nat orb_sym =', orb_sym)
    np.save("nat_coeff.npy", nat_coeff)
    np.save("nat_occ.npy", nat_occ)
    np.save("nat_orb_sym.npy", orb_sym)

txed = time.perf_counter()
print("FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("TOTAL TIME  = %20.3f" % (txed - txst))
