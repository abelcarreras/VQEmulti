from openfermionpyscf import prepare_pyscf_molecule
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.symmetry import symmetrize_molecular_orbitals
import matplotlib.pyplot as plt
import numpy as np


def square_h4_mol(distance, basis='sto-3g'):
    mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                  ['H', [distance, 0, 0]],
                                  ['H', [0, distance, 0]],
                                  ['H', [distance, distance, 0]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='H4')
    return mol


def water_mol(distance=1.0, angle=104.5, basis='6-21g'):
    r = distance
    alpha = np.deg2rad(angle)
    mol = MolecularData(geometry=[['O', [0, 0, 0]],
                                  ['H', [-r, 0, 0]],
                                  ['H', [r*np.cos(np.pi - alpha), r*np.sin(np.pi - alpha), 0]]],
                        basis=basis,
                        multiplicity=2,
                        charge=0,
                        description='H2O')
    return mol


pyscf_molecule = square_h4_mol(distance=1.0, basis='6-31g')
point_group = 'c2h'

# pyscf_molecule = water_mol(distance=2.0, basis='3-21g')

# run classical calculation
molecule = run_pyscf(pyscf_molecule, run_fci=True, run_ccsd=True, nat_orb=False, guess_mix=False, verbose=True)

# get FullCI density matrix in natural orbitals basis (for fidelity calculation)
mo_coefficients = molecule.canonical_orbitals
mo_energies = molecule.orbital_energies

mo_coefficients_sym = symmetrize_molecular_orbitals(mo_coefficients, molecule, point_group, skip=False)

mol = prepare_pyscf_molecule(pyscf_molecule)
n_electrons = pyscf_molecule.n_electrons
n_orbitals = pyscf_molecule.n_orbitals

# fullCI
from pyscf import fci
pyscf_fci = fci.FCI(mol, mo_coefficients, singlet=False)
pyscf_fci.verbose = 0
# pyscf_fci.spin = 0 # s2 s(s+1)
# fci.addons.fix_spin_(pyscf_fci, shift=0.1, ss=0)  # s2
molecule.fci_energy, ci = pyscf_fci.kernel()
# print(ci)


print('FCI energy for {} ({} electrons) is {}.'.format(molecule.name, molecule.n_electrons, molecule.fci_energy))
print('E = %.12f  2S+1 = %.7f' %
      (molecule.fci_energy, pyscf_fci.spin_square(ci, molecule.n_orbitals_active, molecule.n_electrons)[1]))


# Set molecular orbitals basis
rhf_calc = mol.RHF()  # .run()
rhf_calc.mo_coeff = mo_coefficients_sym
rhf_calc.mo_energy = mo_energies
rhf_calc.mo_occ = np.array([2.] * (n_electrons // 2) + [0.] * (n_orbitals - n_electrons // 2))
rhf_calc.converged = True
print('e_tot', rhf_calc.energy_tot())


casci_list = []
cisd_list = []
ccsd_list = []

casci_list_ref = []
cisd_list_ref = []
ccsd_list_ref = []

casci_diff = []
cisd_diff = []
ccsd_diff = []

range_orbitals = list(range(n_electrons//2, n_orbitals+1))
for i_orb in range_orbitals:

    print('\n========', i_orb, '========')
    print('n_elect', n_electrons)
    print('n_orbitals', n_orbitals)
    print('included orb', i_orb)


    # ref calculation
    rhf_calc_ref = mol.RHF() #.run()

    rhf_calc_ref.mo_coeff = mo_coefficients
    rhf_calc_ref.mo_energy = mo_energies
    rhf_calc_ref.mo_occ = np.array([2.] * (n_electrons//2) + [0.] * (n_orbitals - n_electrons//2))
    rhf_calc_ref.converged = True

    # CAS calculation
    cas_calc = rhf_calc.CASCI(i_orb, n_electrons)
    # cas_calc = cas_calc.state_average([1./i_orb] * i_orb)
    #cas_calc.fcisolver.spin = 0
    cas_calc.fix_spin_(shift=0.1, ss=0)


    e_tot, e_cas, other, ci_cas, fcivec = cas_calc.kernel()
    print('e_tot: ', e_tot)

    print('e_cas: ', e_cas)
    print('fcivec: ', fcivec)
    ci0 = np.zeros((28, 28))
    ci0[:len(ci_cas), :len(ci_cas)] = ci_cas[:, :]
    print('*E = %.12f  2S+1 = %.7f' %
          (molecule.fci_energy, pyscf_fci.spin_square(ci, molecule.n_orbitals_active, molecule.n_electrons)[1]))

    from pyscf.fci.spin_op import spin_square
    data = spin_square(ci0, molecule.n_orbitals_active, molecule.n_electrons, mo_coeff=mo_coefficients,
                       fcisolver=cas_calc.fcisolver)
    print('pE = %.12f  2S+1 = %.7f' %
          (e_tot, data[1]))


    # spin_square(fcivec, norb, nelec, mo_coeff=None, ovlp=1, fcisolver=None)


    # print('shape:',len(ci))
    # exit()


    cas_calc_ref = rhf_calc_ref.CASCI(i_orb, n_electrons)
    #cas_calc_ref.fcisolver.spin = 0
    cas_calc_ref.fix_spin_(shift=0.1, ss=0)
    cas_calc_ref.kernel()

    print('CAS: ', i_orb, n_electrons)
    print('Energy CASCI: {:.8f}'.format(cas_calc.e_tot))
    print('Error CASCI: {:.8f}'.format(cas_calc.e_tot - molecule.fci_energy))
    casci_list.append(cas_calc.e_tot - molecule.fci_energy)
    casci_list_ref.append(cas_calc_ref.e_tot - molecule.fci_energy)
    casci_diff.append(cas_calc.e_tot - cas_calc_ref.e_tot)

    # CISD calculation
    list_orb = list(range(i_orb, n_orbitals))
    print('freeze: ', list_orb)
    if len(list_orb) == 0:
        list_orb = None

    cisd_calc = rhf_calc.CISD(frozen=list_orb)
    cisd_calc.kernel()
    cisd_calc_ref = rhf_calc_ref.CISD(frozen=list_orb)
    cisd_calc_ref.kernel()

    print('Energy CISD: {:.8f}'.format(cisd_calc.e_tot))
    print('Error CISD: {:.8f}'.format(cisd_calc.e_tot - molecule.fci_energy))
    print('Correlation CISD: {:.8f}'.format(cisd_calc.e_corr))
    cisd_list.append(cisd_calc.e_tot - molecule.fci_energy)
    cisd_list_ref.append(cisd_calc_ref.e_tot - molecule.fci_energy)
    cisd_diff.append(cisd_calc.e_tot - cisd_calc_ref.e_tot)

    # CCSD calculation
    ccsd_calc = rhf_calc.CCSD(frozen=list_orb)
    ccsd_calc.kernel()
    ccsd_calc_ref = rhf_calc_ref.CCSD(frozen=list_orb)
    ccsd_calc_ref.kernel()

    print('Energy CCSD: {:.8f}'.format(ccsd_calc.e_tot))
    print('Error CCSD: {:.8f}'.format(ccsd_calc.e_tot - molecule.fci_energy))
    print('Correlation CCSD: {:.8f}'.format(ccsd_calc.e_corr))
    ccsd_list.append(ccsd_calc.e_tot - molecule.fci_energy)
    ccsd_list_ref.append(ccsd_calc_ref.e_tot - molecule.fci_energy)
    ccsd_diff.append(ccsd_calc.e_tot - ccsd_calc_ref.e_tot)


plt.title('Error respect to FCI')
plt.plot(range_orbitals, casci_list, '--', label='CASCI (sym)', color='b')
plt.plot(range_orbitals, casci_list_ref, label='CASCI (ref)', color='b')
plt.plot(range_orbitals, cisd_list, '--', label='CISD (sym)', color='r')
plt.plot(range_orbitals, cisd_list_ref, label='CISD (ref)', color='r')
plt.plot(range_orbitals, ccsd_list, '--', label='CCSD (sym)', color='g')
plt.plot(range_orbitals, ccsd_list_ref, label='CCSD (ref)', color='g')

plt.axhline(y=molecule.hf_energy-molecule.fci_energy, color='black', linestyle='--')
plt.axhline(y=rhf_calc.energy_tot()-molecule.fci_energy, color='black', linestyle=':')

plt.ylim(0, None)
plt.xlabel('# orbitals in active space')
plt.ylabel('Error respect to FCI (H)')
plt.legend()


plt.figure()
plt.title('Error diff')
plt.plot(range_orbitals, casci_diff, label='CASCI', color='b')
plt.plot(range_orbitals, cisd_diff, label='CISD', color='r')
plt.plot(range_orbitals, ccsd_diff, label='CCSD', color='g')

plt.xlabel('# orbitals in active space')
plt.ylabel('Error diff (H)')
plt.legend()

plt.show()
