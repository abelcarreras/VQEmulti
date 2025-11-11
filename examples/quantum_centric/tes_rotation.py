import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.linalg import logm, expm


class FermionBasisOperator:
    """
    Construct basis transformation operators as fermion operators.
    """

    def __init__(self, n_orbitals):
        """
        Initialize for n_orbitals (number of spin-orbitals).

        Parameters:
        -----------
        n_orbitals : int
            Number of spin-orbitals
        """
        self.n_orbitals = n_orbitals
        self.dim = 2 ** n_orbitals

    def orbital_rotation_operator(self, U_rotation):
        """
        Create fermion operator for orbital rotation defined by unitary matrix U.

        The fermion operator is: Û = exp(Σ_{p<q} κ_{pq} (a†_p a_q - a†_q a_p))
        where κ is the anti-Hermitian generator: U = exp(κ)

        Parameters:
        -----------
        U_rotation : ndarray (n_orbitals × n_orbitals)
            Unitary orbital rotation matrix

        Returns:
        --------
        fermion_op : dict
            Fermion operator as dictionary: {fermion_string: coefficient}
            e.g., {((0,1), (1,0)): 0.5} represents 0.5 * (a†_0 a_1)
        """
        # Extract the anti-Hermitian generator: κ = log(U)
        kappa = logm(U_rotation)

        # Make sure it's anti-Hermitian (numerically)
        kappa = (kappa - kappa.conj().T) / 2

        fermion_op = {}

        # Build operator: Σ_{p,q} κ_{pq} a†_p a_q
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                if np.abs(kappa[p, q]) > 1e-12:
                    # a†_p a_q is represented as ((p, 1), (q, 0))
                    # where 1 = creation, 0 = annihilation
                    term = ((p, 1), (q, 0))
                    coeff = kappa[p, q]

                    if term in fermion_op:
                        fermion_op[term] += coeff
                    else:
                        fermion_op[term] = coeff

        # For the exponential exp(κ̂), we need to expand or use matrix exponential
        # Return the generator (for small κ) or compute matrix exponential
        return fermion_op

    def orbital_rotation_generator(self, U_rotation):
        """
        Return the generator κ̂ for orbital rotation, not the full exponential.
        This is useful for small rotations or when you want to exponentiate separately.

        Parameters:
        -----------
        U_rotation : ndarray (n_orbitals × n_orbitals)
            Unitary orbital rotation matrix

        Returns:
        --------
        generator_op : dict
            Generator operator κ̂ = Σ_{p,q} κ_{pq} (a†_p a_q)
        """
        kappa = logm(U_rotation)
        kappa = (kappa - kappa.conj().T) / 2

        generator_op = {}

        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                if np.abs(kappa[p, q]) > 1e-12:
                    term = ((p, 1), (q, 0))
                    generator_op[term] = kappa[p, q]

        return generator_op

    def number_operator(self, orbital_idx):
        """
        Create number operator n̂_p = a†_p a_p for orbital p.

        Parameters:
        -----------
        orbital_idx : int
            Orbital index

        Returns:
        --------
        fermion_op : dict
            Number operator as fermion operator
        """
        term = ((orbital_idx, 1), (orbital_idx, 0))
        return {term: 1.0}

    def particle_number_operator(self):
        """
        Create total particle number operator: N̂ = Σ_p a†_p a_p

        Returns:
        --------
        fermion_op : dict
            Total number operator
        """
        number_op = {}
        for p in range(self.n_orbitals):
            term = ((p, 1), (p, 0))
            number_op[term] = 1.0
        return number_op

    def spin_operator(self, component='z'):
        """
        Create spin operator Ŝ_z, Ŝ_+, or Ŝ_-
        Assumes even indices are alpha, odd indices are beta.

        Parameters:
        -----------
        component : str
            'z' for Ŝ_z, '+' for Ŝ_+, '-' for Ŝ_-

        Returns:
        --------
        fermion_op : dict
            Spin operator
        """
        if component == 'z':
            # Ŝ_z = (1/2) Σ_p (n_p^α - n_p^β)
            spin_op = {}
            for p in range(0, self.n_orbitals, 2):  # alpha orbitals
                term = ((p, 1), (p, 0))
                spin_op[term] = 0.5
            for p in range(1, self.n_orbitals, 2):  # beta orbitals
                term = ((p, 1), (p, 0))
                spin_op[term] = -0.5
            return spin_op
        elif component == '+':
            # Ŝ_+ = Σ_p a†_p^α a_p^β
            spin_op = {}
            for p in range(0, self.n_orbitals // 2):
                alpha_idx = 2 * p
                beta_idx = 2 * p + 1
                term = ((alpha_idx, 1), (beta_idx, 0))
                spin_op[term] = 1.0
            return spin_op
        elif component == '-':
            # Ŝ_- = Σ_p a†_p^β a_p^α
            spin_op = {}
            for p in range(0, self.n_orbitals // 2):
                alpha_idx = 2 * p
                beta_idx = 2 * p + 1
                term = ((beta_idx, 1), (alpha_idx, 0))
                spin_op[term] = 1.0
            return spin_op
        else:
            raise ValueError(f"Unknown component: {component}")

    def fermion_op_to_string(self, fermion_op):
        """
        Convert fermion operator dict to human-readable string.

        Parameters:
        -----------
        fermion_op : dict
            Fermion operator

        Returns:
        --------
        str : readable representation
        """
        terms = []
        for operators, coeff in fermion_op.items():
            op_str = ""
            for idx, dag in operators:
                if dag == 1:
                    op_str += f"a†_{idx} "
                else:
                    op_str += f"a_{idx} "
            terms.append(f"({coeff:.4f}) {op_str}")
        return " + ".join(terms)


def example_orbital_rotation():
    """
    Example: Create fermion operator for a simple orbital rotation.
    """
    n_orbitals = 4  # 2 spatial × 2 spin
    fbo = FermionBasisOperator(n_orbitals)

    # Simple rotation: mix orbitals 0 and 2 (both alpha)
    theta = np.pi / 4
    U = np.eye(n_orbitals, dtype=complex)
    U[0, 0] = np.cos(theta)
    U[0, 2] = -np.sin(theta)
    U[2, 0] = np.sin(theta)
    U[2, 2] = np.cos(theta)

    print("Rotation matrix U:")
    print(U)

    # Get generator
    generator = fbo.orbital_rotation_generator(U)

    print("\nGenerator κ̂ as fermion operator:")
    print(fbo.fermion_op_to_string(generator))

    print("\nFermion operator terms:")
    for term, coeff in generator.items():
        print(f"  {term}: {coeff:.6f}")

    return generator


def example_number_and_spin_operators():
    """
    Example: Create various fermion operators.
    """
    n_orbitals = 4
    fbo = FermionBasisOperator(n_orbitals)

    # Number operator for orbital 0
    n0 = fbo.number_operator(0)
    print("Number operator n̂_0:")
    print(fbo.fermion_op_to_string(n0))

    # Total number operator
    N = fbo.particle_number_operator()
    print("\nTotal number operator N̂:")
    print(fbo.fermion_op_to_string(N))

    # Spin operators
    Sz = fbo.spin_operator('z')
    print("\nSpin Ŝ_z operator:")
    print(fbo.fermion_op_to_string(Sz))

    Splus = fbo.spin_operator('+')
    print("\nSpin Ŝ_+ operator:")
    print(fbo.fermion_op_to_string(Splus))


if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Orbital Rotation Generator")
    print("=" * 70)
    example_orbital_rotation()

    print("\n" + "=" * 70)
    print("Example 2: Number and Spin Operators")
    print("=" * 70)
    example_number_and_spin_operators()

    print("\n" + "=" * 70)
    print("Note: These fermion operators can be converted to sparse matrices")
    print("using your existing Jordan-Wigner or other encoding functions.")
    print("=" * 70)