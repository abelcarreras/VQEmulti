from openfermion import MolecularData
from pyscf import scf, fci
from pyscf import gto
from vqemulti.utils import fermion_to_qubit
"""
 Class for computing atomic energies using
 RHF
 KS-DFT
 FCI
 
 Also computes the Hamiltonian in qubit form for use in VQEmulti.
"""
def separate_identity_term(qubit_operator):
    from openfermion import QubitOperator
    result = QubitOperator()

    constant = 0
    for term, coefficient in qubit_operator.terms.items():
        if term:
            result += QubitOperator(term, coefficient)
        else:
            constant += coefficient

    return result, constant

class AtomSystem:
    def __init__(self, atoms, charge, spin=0, basis="sto-3g"):
        """
        :param atoms: List of tuples, e.g. [("O", (0,0,0)),]
        :param basis: string defining the basis set (default 'sto3g')
        :param charge: the charge of the atom
        :param spin: 2S where 0-for closed-shell singlets
        """
        self.atoms = atoms
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.mol = self._build_mol()
        self.results = {}
        self.n_electrons = 0
        self.n_orbitals = 0
        self.n_qubits = 0
        self.H_terms = None
        #self.FermionicHamiltonian = None
        #self.QubitHamiltonian = None


    def _build_mol(self):
        return gto.Mole(
            atom=self.atoms,
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            verbose=0,
        ).build()

    def run_rhf(self):
        mf = scf.RHF(mol=self.mol)
        energy = mf.kernel()
        self.results['RHF'] = (energy, mf)
        return energy

    def run_dft(self, functional="b3lyp"):
        mf = scf.RKS(mol=self.mol)
        mf.xc = functional
        energy = mf.kernel()
        self.results['DFT'] = (energy, mf)
        return energy

    def run_fci(self):
        if 'RHF' not in self.results:
            self.run_rhf()
        hf_energy, mf = self.results['RHF']
        cisolver = fci.FCI(mf)
        e_fci, _ = cisolver.kernel()
        self.results['FCI'] = (e_fci, cisolver)
        return e_fci

    def compare_energies(self):
        """
        Return dictionary of correlation energies:
        E_corr = E_method - E_FCI
        """
        if "RHF" not in self.results:
            raise ValueError("Run RHF first...")
        hf_energy, mf = self.results['RHF']
        comparison = {}

        for method in ["DFT","FCI"]:
            if method in self.results:
                e, _ = self.results[method]
                comparison[method] = e - hf_energy
        return comparison

    def get_hamiltonian(self):
        from openfermionpyscf import run_pyscf
        from openfermion import MolecularData
        # run reference calculation

        atom = MolecularData(geometry=self.atoms,
                         basis=self.basis,
                         multiplicity=self.spin +1, #spin = 2S, multiplicity = 2S+1
                         charge=self.charge,
                         description='atom')

        # frozen_core=?, n_orbitals=? are disabled in this version
        molecule = run_pyscf(atom,
                             run_fci=False, nat_orb=False,
                             guess_mix=False, verbose=True,
                             run_ccsd=False, run_casci=True)

        # get hamiltonian
        hamiltonian = molecule.get_molecular_hamiltonian()
        hamiltonian_te = fermion_to_qubit(hamiltonian)
        hamiltonian_te, constant = separate_identity_term(hamiltonian_te)

        # Store information
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals
        self.n_qubits = molecule.n_qubits
        self.H_terms = hamiltonian_te.terms
        #self.FermionicHamiltonian = hamiltonian
        #self.QubitHamiltonian = hamiltonian_te

        return hamiltonian_te, constant
