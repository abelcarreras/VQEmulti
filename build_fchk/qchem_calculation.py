# Simple single point calculation
from pyqchem import get_output_from_qchem, QchemInput, Structure
from pyqchem.parsers.basic import basic_parser_qchem
from pyqchem.file_io import write_to_fchk
import numpy as np

r = 1.5
molecule = Structure(coordinates=[[0.0, 0.0, 0.0],
                                  [0.0, 0.0, r],
                                  [0.0, 0.0, 2*r],
                                  [0.0, 0.0, 3*r]],
                     symbols=['H', 'H', 'H', 'H'],
                     charge=0,
                     multiplicity=1)


# create qchem input
qc_input = QchemInput(molecule,
                      jobtype='sp',
                      exchange='hf',
                      basis='3-21g',
                      unrestricted=False)

# calculate and parse qchem output

data, ee = get_output_from_qchem(qc_input,
                                 processors=4,
                                 force_recalculation=True,
                                 parser=basic_parser_qchem,
                                 return_electronic_structure=True)

write_to_fchk(ee, 'qchem_calc.fchk')


print('scf energy', data['scf_energy'], 'H')

print('Orbital energies (H)')
print('  {:^10} {:^10}'.format('alpha', 'beta'))
for a, b in zip(data['orbital_energies']['alpha'], data['orbital_energies']['beta']):
    print('{:10.3f} {:10.3f}'.format(a, b))

print('dipole moment:', data['multipole']['dipole_moment'], data['multipole']['dipole_units'])
print('quadrupole moment ({}):\n'.format(data['multipole']['quadrupole_units']),
      np.array(data['multipole']['quadrupole_moment']))
print('octopole moment ({}):\n'.format(data['multipole']['octopole_units']),
      np.array(data['multipole']['octopole_moment']))
