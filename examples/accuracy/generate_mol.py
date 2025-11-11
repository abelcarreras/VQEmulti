import math
from openfermion import MolecularData


def generate_tetrahedron_coordinates(axis_length):
    # Calculate the height of the tetrahedron
    height = math.sqrt(2/3) * axis_length

    # Calculate the coordinates of the vertices
    vertices = []
    vertices.append((0, 0, 0))
    vertices.append((axis_length, 0, 0))
    vertices.append((axis_length/2, axis_length*math.sqrt(3)/2, 0))
    vertices.append((axis_length/2, axis_length*math.sqrt(3)/6, height))

    return vertices


def tetra_h4_mol(distance, basis='sto-3g'):
    coor = generate_tetrahedron_coordinates(distance)

    mol = MolecularData(geometry=[['H', coor[0]],
                                  ['H', coor[1]],
                                  ['H', coor[2]],
                                  ['H', coor[3]]],
                        basis=basis,
                        multiplicity=1,
                        charge=2,
                        description='H4')
    return mol


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


def linear_h4_mol(distance, basis='sto-3g'):
    mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                  ['H', [0, 0, distance]],
                                  ['H', [0, 0, 2 * distance]],
                                  ['H', [0, 0, 3 * distance]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='H4')
    return mol


def h2_mol(distance, basis='sto-3g'):
    mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                  ['H', [0, 0, distance]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='H2')
    return mol



if __name__ == '__main__':
    mol = tetra_h4_mol(distance=4.3)
    print(len(mol.geometry), '\n')
    for c in mol.geometry:
        print('{:3} {:10.5f} {:10.5f} {:10.5f}'.format(c[0], c[1][0], c[1][1], c[1][2]))


