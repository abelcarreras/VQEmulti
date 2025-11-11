import numpy as np
from pyscf import gto, scf, lo
from pyscf.tools import cubegen

x = .63
mol = gto.M(atom=[['C', (0, 0, 0)],
                  ['H', (x ,  x,  x)],
                  ['H', (-x, -x,  x)],
                  ['H', (-x,  x, -x)],
                  ['H', ( x, -x, -x)]],
            basis='ccpvtz')
mf = scf.RHF(mol).run()

# C matrix stores the AO to localized orbital coefficients
C = lo.orth_ao(mf, 'nao')
orbs = C[:,mf.mo_occ>0] # Only get occupied orbitals

for i in range(orbs.shape[1]):
    cubegen.orbital(mol, f'ch4_nao_mo{i+1}.cube', orbs[:,i])