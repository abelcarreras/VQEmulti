# example of hydrogen molecule using UCCSD ansatz with VQE method
import matplotlib.pyplot as plt

if False:
    from vqemulti.utils import get_hf_reference_in_fock_space
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from vqemulti import vqe
    import numpy as np
    from vqemulti.utils import generate_reduced_hamiltonian, get_uccsd_operators
    from pyscf import gto


    n_points = 40
    hf_energies = []
    uhf_energies = []
    for d in np.linspace(0.3, 3, n_points):

        mol = gto.Mole()
        mol.build(
            atom='H  0 0 0;  H 0 0 {}'.format(d),
            basis='sto-3g',
        )

        from pyscf import scf

        mol.charge = 0
        mol.spin = 0      # 2j == nelec_alpha - nelec_beta
        mol.symmetry = 0  # Allow the program to apply point group symmetry if possible
        mol.unit = 'Ang'  # (New in version 1.1)

        hf_h2 = scf.HF(mol)
        hf_h2.silent = True
        hf_h2.kernel()

        hf_energies.append(hf_h2.energy_tot())

        # uhf_h2 = scf.UHF(mol).run(dm_init, conv_tol=1e-5)

        dm_init = np.array([[0.9, 0.1],
                            [0.2, 0.8]])


        # uhf_h2 = scf.UHF(mol).run(dm_init, conv_tol=1e-8)
        uhf_h2 = scf.UHF(mol).run()
        occ = [[1, 0], [0.7, 0.3]]
        mo_coeff = uhf_h2.mo_coeff
        dm1 = uhf_h2.make_rdm1(mo_coeff, occ)


        #mo_i, mo_e = uhf_h2.stability()

        #print(uhf_h2.mo_occ)
        #dm1 = uhf_h2.make_rdm1(mo_i, occ)
        uhf_h2 = uhf_h2.run(dm1, conv_tol=1e-18)

        # uhf_h2.stability()

        #uhf_h2 = uhf_h2.newton().run(mo_i, uhf_h2.mo_occ)
        #uhf_h2.stability()

        #uhf_h2.stability(external=True)

        uhf_energies.append(uhf_h2.energy_tot())

    plt.title('Absolute energies')
    plt.xlabel('Interatomic distance [Angs]')
    plt.ylabel('Energy [H]')

    plt.plot(np.linspace(0.3, 3, n_points), hf_energies, label='HF')
    plt.plot(np.linspace(0.3, 3, n_points), uhf_energies, '--', label='UHF')

    plt.legend()

    plt.show()

import numpy
import pyscf
from pyscf import scf
from pyscf import gto

erhf = []
euhf = []
dm = None


def init_guess_mixed(mol, mixing_parameter=numpy.pi / 4):
    ''' Generate density matrix with broken spatial and spin symmetry by mixing
    HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.

    psi_1a = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
    psi_1b = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo

    psi_2a = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
    psi_2b =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo

    Returns:
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    # opt: q, mixing parameter 0 < q < 2 pi

    # based on init_guess_by_1e
    h1e = scf.hf.get_hcore(mol)
    s1e = scf.hf.get_ovlp(mol)
    mo_energy, mo_coeff = rhf.eig(h1e, s1e)
    mo_occ = rhf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    homo_idx = 0
    lumo_idx = 1

    for i in range(len(mo_occ) - 1):
        if mo_occ[i] > 0 and mo_occ[i + 1] < 0:
            homo_idx = i
            lumo_idx = i + 1

    psi_homo = mo_coeff[:, homo_idx]
    psi_lumo = mo_coeff[:, lumo_idx]

    Ca = numpy.zeros_like(mo_coeff)
    Cb = numpy.zeros_like(mo_coeff)

    # mix homo and lumo of alpha and beta coefficients
    q = mixing_parameter

    for k in range(mo_coeff.shape[0]):
        if k == homo_idx:
            Ca[:, k] = numpy.cos(q) * psi_homo + numpy.sin(q) * psi_lumo
            Cb[:, k] = numpy.cos(q) * psi_homo - numpy.sin(q) * psi_lumo
            continue
        if k == lumo_idx:
            Ca[:, k] = -numpy.sin(q) * psi_homo + numpy.cos(q) * psi_lumo
            Cb[:, k] = numpy.sin(q) * psi_homo + numpy.cos(q) * psi_lumo
            continue
        Ca[:, k] = mo_coeff[:, k]
        Cb[:, k] = mo_coeff[:, k]

    dm = scf.UHF(mol).make_rdm1((Ca, Cb), (mo_occ, mo_occ))
    return dm


for b in numpy.arange(0.3, 4.01, 0.1):
    mol = gto.M(atom=[["H", 0., 0., 0.],
                      ["H", 0., 0., b]], basis='3-21g', verbose=0)
    rhf = scf.RHF(mol)
    uhf = scf.UHF(mol)
    erhf.append(rhf.kernel(dm))
    euhf.append(uhf.kernel(init_guess_mixed(mol)))
    dm = rhf.make_rdm1()

print('R     E(RHF)      E(UHF)')
for i, b in enumerate(numpy.arange(0.7, 4.01, 0.1)):
    print('%.2f  %.8f  %.8f' % (b, erhf[i], euhf[i]))


plt.title('Absolute energies')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')

plt.plot(numpy.arange(0.3, 4.01, 0.1), erhf, label='HF')
plt.plot(numpy.arange(0.3, 4.01, 0.1), euhf, '--', label='UHF')

plt.legend()

plt.show()

