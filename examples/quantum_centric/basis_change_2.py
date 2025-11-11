import numpy as np
import scipy as sp
from openfermion import FermionOperator, hermitian_conjugated, get_sparse_operator
from scipy.linalg import logm


def get_sparse_U(U_test):
    """
    Convert spin-orbital rotation matrix to sparse second quantized representation.
    """
    n_spin_orbitals = U_test.shape[0]

    # Get the generator of the rotation (anti-hermitian matrix)
    theta_matrix = logm(U_test)

    # Build the second quantized rotation operator
    rotation_op = FermionOperator()

    for p in range(n_spin_orbitals):
        for q in range(n_spin_orbitals):
            if abs(theta_matrix[p, q]) > 1e-12:
                # Single excitation generator: a_p† a_q - a_q† a_p
                rotation_op += theta_matrix[p, q] * (
                        FermionOperator(f'{p}^ {q}') -
                        FermionOperator(f'{q}^ {p}')
                )

    # Convert to sparse matrix and exponentiate
    sparse_rotation = get_sparse_operator(rotation_op)
    sparse_U = sp.sparse.linalg.expm(sparse_rotation)

    return sparse_U


def get_sparse_cc(t1, t2=None):
    """
    Build the CC operator with T1 and optionally T2 amplitudes
    """
    operator_tot = FermionOperator()

    # T1 amplitudes (single excitations)
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            if abs(t1[i, j]) > 1e-12:
                operator = FermionOperator('{}^ {}'.format(i, j))
                operator_tot += t1[i, j] * (operator - hermitian_conjugated(operator))

    # T2 amplitudes (double excitations) - if provided
    if t2 is not None:
        for i in range(t2.shape[0]):
            for j in range(t2.shape[1]):
                for k in range(t2.shape[2]):
                    for l in range(t2.shape[3]):
                        if abs(t2[i, j, k, l]) > 1e-12:
                            operator = FermionOperator('{}^ {}^ {} {}'.format(i, j, l, k))
                            operator_tot += t2[i, j, k, l] * (operator - hermitian_conjugated(operator))

    return get_sparse_operator(operator_tot)


def test_transformation_equivalence():
    """
    Test the equivalence of the two transformation methods with a simple example.
    """
    # Create a simple test case
    n_orbitals = 4  # Small system for testing

    # Create a simple T1 amplitude matrix (antisymmetric for testing)
    t1 = np.random.rand(n_orbitals, n_orbitals) * 0.1
    t1 = t1 - t1.T  # Make antisymmetric

    # Create a simple unitary transformation (close to identity)
    angle = 0.1
    U_test = np.eye(n_orbitals) + angle * (np.random.rand(n_orbitals, n_orbitals) - 0.5)
    U_test = U_test - U_test.T  # Make antisymmetric
    U_test = sp.linalg.expm(U_test)  # Make unitary

    # Zero T2 for simplicity
    t2 = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))

    # Create a simple reference state (all electrons in first n_orbitals/2 orbitals)

    # Method 1: Transform amplitudes first
    print("=== Method 1: Transform amplitudes first ===")
    bc_single_amps = U_test @ t1 @ U_test.T
    sparse_cc_op_1 = get_sparse_cc(bc_single_amps, t2)

    # Method 2: Transform operator
    print("=== Method 2: Transform operator ===")
    sparse_U = get_sparse_U(U_test)
    sparse_cc_op_2 = get_sparse_cc(t1, t2)
    sparse_cc_op_2_transformed = sparse_U.getH() @ sparse_cc_op_2 @ sparse_U

    # Check if operators are equivalent
    op1_dense = sparse_cc_op_1.toarray()
    op2_dense = sparse_cc_op_2_transformed.toarray()

    print(f"Operator 1 shape: {op1_dense.shape}")
    print(f"Operator 2 shape: {op2_dense.shape}")
    print(f"Max difference in operators: {np.max(np.abs(op1_dense - op2_dense))}")
    print(f"Relative difference: {np.max(np.abs(op1_dense - op2_dense)) / np.max(np.abs(op1_dense))}")

    # Additional checks
    print(f"\n=== Debugging Info ===")
    print(f"U_test is unitary: {np.allclose(U_test @ U_test.T.conj(), np.eye(n_orbitals))}")
    print(f"U_test determinant: {np.linalg.det(U_test)}")

    sparse_U_dense = sparse_U.toarray()
    unitarity_check = sparse_U_dense @ sparse_U_dense.T.conj()
    unitarity_error = np.max(np.abs(unitarity_check - np.eye(sparse_U_dense.shape[0])))
    print(f"sparse_U unitarity error: {unitarity_error}")

    print(f"Original t1 max: {np.max(np.abs(t1))}")
    print(f"Transformed t1 max: {np.max(np.abs(bc_single_amps))}")

    return np.max(np.abs(op1_dense - op2_dense))


# Run the test
if __name__ == "__main__":
    max_diff = test_transformation_equivalence()

    if max_diff < 1e-10:
        print(f"\n✓ SUCCESS: Transformations are equivalent (diff = {max_diff:.2e})")
    elif max_diff < 1e-6:
        print(f"\n⚠ WARNING: Small differences detected (diff = {max_diff:.2e})")
    else:
        print(f"\n✗ ERROR: Large differences detected (diff = {max_diff:.2e})")
        print("There may be an implementation issue!")