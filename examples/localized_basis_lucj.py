from openfermion import MolecularData
from openfermionpyscf import run_pyscf, store_orbitals_in_molden
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.ansatz.unitary_jastrow import UnitaryCoupledJastrowAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.ansatz.generators.rotation import change_of_basis_orbitals
from vqemulti.ansatz.generators.basis import get_absolute_orbitals
from vqemulti.preferences import Configuration
from copy import deepcopy


Configuration().mapping = 'jw'
Configuration().verbose = True
Configuration().temp_dir = '/Users/abel/TEST/DICE'


tetracene = MolecularData(
    geometry=[
        ('C', [-4.919,  0.710, 0.0]),
        ('C', [-3.689,  1.420, 0.0]),
        ('C', [-2.459,  0.710, 0.0]),
        ('C', [-1.230,  1.420, 0.0]),
        ('C', [ 0.000,  0.710, 0.0]),
        ('C', [ 1.230,  1.420, 0.0]),
        ('C', [ 2.459,  0.710, 0.0]),
        ('C', [ 3.689,  1.420, 0.0]),
        ('C', [ 4.919,  0.710, 0.0]),

        ('C', [-4.919, -0.710, 0.0]),
        ('C', [-3.689, -1.420, 0.0]),
        ('C', [-2.459, -0.710, 0.0]),
        ('C', [-1.230, -1.420, 0.0]),
        ('C', [ 0.000, -0.710, 0.0]),
        ('C', [ 1.230, -1.420, 0.0]),
        ('C', [ 2.459, -0.710, 0.0]),
        ('C', [ 3.689, -1.420, 0.0]),
        ('C', [ 4.919, -0.710, 0.0]),

        ('H', [-5.874,  1.262, 0.0]),
        ('H', [-5.874, -1.262, 0.0]),
        ('H', [-3.689,  2.524, 0.0]),
        ('H', [-3.689, -2.524, 0.0]),
        ('H', [-1.230,  2.524, 0.0]),
        ('H', [-1.230, -2.524, 0.0]),
        ('H', [ 1.230,  2.524, 0.0]),
        ('H', [ 1.230, -2.524, 0.0]),
        ('H', [ 3.689,  2.524, 0.0]),
        ('H', [ 3.689, -2.524, 0.0]),
        ('H', [ 5.874,  1.262, 0.0]),
        ('H', [ 5.874, -1.262, 0.0]),
    ],
    basis='3-21g',
    multiplicity=1,
    charge=0,
    description='tetracene'
)

# 120 electrons

#active = 6
active = 12

print('frozen: ', 120//2-active//2)

# run reference calculation
molecule_loc = run_pyscf(deepcopy(tetracene),
                         run_fci=False,
                         nat_orb=False,
                         guess_mix=False,
                         verbose=True,
                         frozen_core=120//2-active//2,
                         n_orbitals=120//2+active//2,
                         #run_ccsd=True,
                         run_casci=True,
                         loc_boys=False,
                         loc_pipek=True,
                         reference='HF')


store_orbitals_in_molden(molecule_loc, filename='orbitals.molden')

# Build local hamiltonian
trans_mat = molecule_loc.canonical_local_trans_mat # [120//2-active//2:120//2+active//2, 120//2-active//2:120//2+active//2]
hamiltonian_loc = molecule_loc.get_molecular_hamiltonian()


# hamiltonian data
n_electrons = molecule_loc.n_electrons
n_orbitals = molecule_loc.n_orbitals
multiplicity = molecule_loc.multiplicity

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('multiplicity', multiplicity)
print('n_qubits:', hamiltonian_loc.n_qubits)


molecule = run_pyscf(deepcopy(tetracene),
                     run_fci=False,
                     nat_orb=False,
                     guess_mix=False,
                     verbose=True,
                     frozen_core=120//2-active//2,
                     n_orbitals=120//2+active//2,
                     run_ccsd=True,
                     run_casci=True,
                     loc_boys=False,
                     loc_pipek=False,
                     reference='HF')


hamiltonian = molecule.get_molecular_hamiltonian()

# scale t1 to be noticeable (unphysical)
ccsd = molecule._pyscf_data.get('ccsd', None)
ccsd.t1 = ccsd.t1 * 1000


# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian_loc.n_qubits, multiplicity)
print('hf reference', hf_reference_fock)

# Build LUCJ ansatz using canonical basis
ucja_ansatz = UnitaryCoupledJastrowAnsatz(ccsd.t1,
                                          ccsd.t2,
                                          hf_reference_fock,
                                          n_terms=1,
                                          full_trotter=False,
                                          local=None)


energy = ucja_ansatz.get_energy(ucja_ansatz.parameters, hamiltonian, None)
print('LUCJ energy canonical: ', energy)


# Build LUCJ ansatz local
ccsd = molecule._pyscf_data.get('ccsd', None)
T1_orb = get_absolute_orbitals(ccsd.t1)  # a_j a_i^ -> a_j  a_i^
T2_orb = get_absolute_orbitals(ccsd.t2)  # a_j a_l a_i^ a_k^ ->  a_j a_l a_i^ a_k^


# Build LUCJ ansatz using canonical basis with absolute amplitudes
ucja_ansatz = UnitaryCoupledJastrowAnsatz(T1_orb,
                                          T2_orb,
                                          hf_reference_fock,
                                          n_terms=1,
                                          full_trotter=False,
                                          local=None,
                                          use_general=True)

energy = ucja_ansatz.get_energy(ucja_ansatz.parameters, hamiltonian, None)
print('LUCJ energy canonical abs: ', energy)




# generate amplitudes in the local basis
T1_orb_loc, T2_orb_loc = change_of_basis_orbitals(T1_orb, T2_orb, trans_mat)

# Build LUCJ ansatz using local basis with absolute amplitudes
ucja_ansatz_loc = UnitaryCoupledJastrowAnsatz(T1_orb_loc,
                                              T2_orb_loc,
                                              hf_reference_fock,
                                              n_terms=1,
                                              full_trotter=False,
                                              local=None,
                                              use_general=True,
                                              reference_basis=trans_mat)

energy_loc = ucja_ansatz_loc.get_energy(ucja_ansatz_loc.parameters, hamiltonian_loc, None)
print('LUCJ energy localized abs:', energy_loc)


simulator = Simulator(trotter=False,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000000)

# compute energy in the simulator
energy_loc = ucja_ansatz_loc.get_energy(ucja_ansatz_loc.parameters, hamiltonian_loc, simulator)
print('LUCJ energy localized sim:', energy_loc)


