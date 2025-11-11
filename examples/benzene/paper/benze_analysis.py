from pyscf import gto, scf, grad
from scipy.optimize import minimize
import numpy as np

def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()[2:]
    atoms = []
    coords = []
    for line in lines:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)

atoms, coords = read_xyz("benzene.xyz")

def build_mol(coords_flat):
    mol = gto.Mole()
    mol.unit = 'Angstrom'
    mol.atom = [(atoms[i], coords_flat[3*i:3*i+3]) for i in range(len(atoms))]
    mol.basis = 'sto-3g'
    mol.build()
    return mol

def energy(coords_flat):
    mol = build_mol(coords_flat)
    mf = scf.RHF(mol)
    return mf.kernel()

# Inicialització
x0 = coords.flatten()

# Optimització geomètrica
res = minimize(energy, x0, method='BFGS', options={'disp': True})

# Resultats
print("Coordenades optimitzades:")
opt_coords = res.x.reshape(-1, 3)
for atom, xyz in zip(atoms, opt_coords):
    print(f"{atom} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")

