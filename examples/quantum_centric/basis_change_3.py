import numpy as np
import scipy as sp
from openfermion import FermionOperator, hermitian_conjugated, get_sparse_operator
from scipy.linalg import logm


def get_sparse_U_orbital_rotation(U_test):
    """
    Create sparse unitary matrix for orbital rotation.
    This creates the unitary that rotates the ORBITALS, not the excitation operators.
    """
    n_spin_orbitals = U_test.shape[0]

    # For orbital rotations, we need the unitary that acts on the fermionic operators
    # The orbital rotation U acts as: a_p† -> sum_q U_pq a_q†
    # This corresponds to the second quantized unitary: exp(sum_pq θ_pq (a_p† a_q - a_q† a_p))

    # Get the generator (anti-hermitian matrix)
    theta_matrix = logm(U_test)

    # Build the second quantized rotation operator
    rotation_op = FermionOperator()

    for p in range(n_spin_orbitals):
        for q in range(n_spin_orbitals):
            if abs(theta_matrix[p, q]) > 1e-12:
                # Generator: a_p† a_q - a_q† a_p
                rotation_op += theta_matrix[p, q] * (
                        FermionOperator(f'{p}^ {q}') -
                        FermionOperator(f'{q}^ {p}')
                )

    # Convert to sparse matrix and exponentiate
    sparse_rotation = get_sparse_operator(rotation_op, n_qubits=len(U_test))

    print(f"Rotation operator shape: {sparse_rotation.shape}")
    print(f"Expected Fock space dim: {2 ** n_spin_orbitals}")

    sparse_U = sp.sparse.linalg.expm(sparse_rotation)

    return sparse_U


def get_sparse_U_direct(U_test):
    """
    Alternative approach: Build the unitary transformation directly from the orbital rotation matrix.
    This constructs the second quantized unitary that corresponds to the orbital rotation.
    """
    n_spin_orbitals = U_test.shape[0]

    # Create the Fock space dimension (2^n for n orbitals)
    fock_dim = 2 ** n_spin_orbitals

    # This is a more complex construction that requires mapping from
    # orbital rotations to Fock space transformations
    # For now, let's use the generator approach but with a different construction

    # Alternative: Build using the relation between orbital and second quantized transformations
    # U_2nd = exp(sum_pq θ_pq (a_p† a_q - a_q† a_p))

    return get_sparse_U_orbital_rotation(U_test)


def get_sparse_cc(t1, t2=None, tolerance=1e-12):
    """
    Build the CC operator with T1 and optionally T2 amplitudes
    """
    operator_tot = FermionOperator()

    # T1 amplitudes (single excitations)
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            if abs(t1[i, j]) > tolerance:
                operator = FermionOperator('{}^ {}'.format(i, j))
                operator_tot += t1[i, j] * (operator - hermitian_conjugated(operator))

    # T2 amplitudes (double excitations) - if provided
    if t2 is not None:
        for i in range(t2.shape[0]):
            for j in range(t2.shape[1]):
                for k in range(t2.shape[2]):
                    for l in range(t2.shape[3]):
                        if abs(t2[i, j, k, l]) > tolerance:
                            operator = FermionOperator('{}^ {}^ {} {}'.format(i, j, l, k))
                            operator_tot += t2[i, j, k, l] * (operator - hermitian_conjugated(operator))

    return get_sparse_operator(operator_tot)



def gram_schmidt(A):
    """Stabilize orthogonality"""
    Q, R = np.linalg.qr(A)
    return Q


def random_rotation_qr(n):
    """Genera matriu de rotació correcta amb QR"""
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)

    # Correcció de fase adequada
    d = np.diag(R)
    ph = d / np.abs(d)  # Signes de la diagonal de R
    Q = Q @ np.diag(ph)

    # Si encara té determinant negatiu, corregeix la última columna
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    return Q

def robust_rotation_test(U_test, tolerance=1e-50):
    """Test que té en compte el conditioning"""

    # Verifica el conditioning
    cond_num = np.linalg.cond(U_test)
    print(cond_num)
    print(f"Condition number: {cond_num:.2e}")

    if cond_num > 1e12:
        print("WARNING: Matrix is ill-conditioned!")
        return False

    # Ajusta la tolerància segons el conditioning
    adjusted_tol = tolerance * cond_num

    # El teu test aquí amb adjusted_tol
    # ...

    return True


def test_simple_case(angle):
    """
    Test with a very simple case to understand the issue.
    """
    print("=== Testing Simple Case ===")

    # Very simple 2-orbital system
    n_orbitals = 2

    # Simple T1 amplitude (antisymmetric)
    t1 = np.array([[ 0.0, 0.1],
                   [-0.1, 0.0]])

    # Simple rotation (small angle)
    angle = 0.549
    U_test = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    print(U_test)
    #exit()
    # general rotation
    U_test = random_rotation_qr(2)
    U_test_ = np.array([[0.85319976, -0.52158429],
                       [0.52158429, -0.85319976]])

    U_test_ = np.array([[0.85304678, -0.52183444],
                       [0.52183444,  0.85304678]])


    if not robust_rotation_test(U_test):
        raise Exception('U_test not robust')


    print(f"U_test:\n{U_test}")
    print(f"U_test @ U_test.T:\n{U_test @ U_test.T}")
    print(f"Original t1:\n{t1}")

    # Method 1: Transform amplitudes
    bc_single_amps = U_test @ t1 @ U_test.T
    print(f"Transformed t1:\n{bc_single_amps}")

    # Build operators
    sparse_cc_op_1 = get_sparse_cc(bc_single_amps)

    # Method 2: Transform operator
    sparse_U = get_sparse_U_orbital_rotation(U_test)
    print('Shape: ', sparse_U.shape)
    sparse_cc_op_2 = get_sparse_cc(t1)
    sparse_cc_op_2_transformed = sparse_U.getH() @ sparse_cc_op_2 @ sparse_U

    # Compare
    op1_dense = sparse_cc_op_1.toarray()
    op2_dense = sparse_cc_op_2_transformed.toarray()

    print(f"\nOperator 1 (Method 1):")
    print(f"Shape: {op1_dense.shape}")
    print(f"Max element: {np.max(np.abs(op1_dense))}")

    print(f"\nOperator 2 (Method 2):")
    print(f"Shape: {op2_dense.shape}")
    print(f"Max element: {np.max(np.abs(op2_dense))}")

    diff = np.max(np.abs(op1_dense - op2_dense))
    print(f"\nMax difference: {diff}")
    print(f"Relative difference: {diff / np.max(np.abs(op1_dense))}")

    sign = U_test[0, 0] * U_test[1, 1] / abs(U_test[0, 0] * U_test[1, 1])
    sign = np.linalg.det(U_test)/abs(np.linalg.det(U_test))

    return diff, sign


def test_identity_case():
    """
    Test with identity transformation - should give zero difference.
    """
    print("\n=== Testing Identity Case ===")

    n_orbitals = 3

    # Random T1 amplitude (antisymmetric)
    t1 = np.random.rand(n_orbitals, n_orbitals) * 0.1
    t1 = t1 - t1.T

    # Identity transformation
    U_identity = np.eye(n_orbitals)

    # Method 1: Transform amplitudes (should be unchanged)
    bc_single_amps = U_identity @ t1 @ U_identity.T
    sparse_cc_op_1 = get_sparse_cc(bc_single_amps)

    # Method 2: Transform operator (should be unchanged)
    sparse_U = get_sparse_U_orbital_rotation(U_identity)
    sparse_cc_op_2 = get_sparse_cc(t1)

    # Debug dimensions
    print(f"t1 shape: {t1.shape}")
    print(f"sparse_cc_op_1 shape: {sparse_cc_op_1.shape}")
    print(f"sparse_cc_op_2 shape: {sparse_cc_op_2.shape}")
    print(f"sparse_U shape: {sparse_U.shape}")

    # Check if dimensions match
    if sparse_U.shape[0] != sparse_cc_op_2.shape[0]:
        print(f"ERROR: Dimension mismatch!")
        print(f"sparse_U: {sparse_U.shape}")
        print(f"sparse_cc_op_2: {sparse_cc_op_2.shape}")
        print(f"Expected Fock space dim for {n_orbitals} orbitals: {2 ** n_orbitals}")
        return float('inf')

    sparse_cc_op_2_transformed = sparse_U.getH() @ sparse_cc_op_2 @ sparse_U

    # Compare
    op1_dense = sparse_cc_op_1.toarray()
    op2_dense = sparse_cc_op_2_transformed.toarray()

    diff = np.max(np.abs(op1_dense - op2_dense))
    print(f"Identity test difference: {diff}")

    # Also check if sparse_U is actually identity
    sparse_U_dense = sparse_U.toarray()
    identity_diff = np.max(np.abs(sparse_U_dense - np.eye(sparse_U_dense.shape[0])))
    print(f"sparse_U vs identity difference: {identity_diff}")

    return diff


if __name__ == "__main__":
    # Test identity case first
    identity_diff = test_identity_case()

    if identity_diff < 1e-10:
        print("✓ Identity test passed")
    else:
        print(f"✗ Identity test failed (diff = {identity_diff:.2e})")

    # Test simple case
    diff_list = []
    sign_list = []
    for angle in np.linspace(0, 2*np.pi, 100):
        simple_diff, sign = test_simple_case(angle)
        diff_list.append(simple_diff)
        sign_list.append(sign)

        if simple_diff < 1e-6:
            print("✓ Simple test passed")
        else:
            print(f"✗ Simple test failed (diff = {simple_diff:.2e})")
        # exit()

    import matplotlib.pyplot as plt
    plt.plot(diff_list)
    plt.plot(sign_list)
    plt.show()