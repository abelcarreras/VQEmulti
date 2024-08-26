from setuptools import setup


def get_version_number():
    main_ns = {}
    for line in open('vqemulti/__init__.py', 'r').readlines():
        if not(line.find('__version__')):
            exec(line, main_ns)
            return main_ns['__version__']


setup(name='vqemulti',
      version=get_version_number(),
      description='implementation of vqe algorithms',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Abel Carreras',
      author_email='abel.carreras@multiversecomputing.com',
      packages=['vqemulti',
                'vqemulti.energy',
                'vqemulti.gradient',
                'vqemulti.pool',
                'vqemulti.simulators',
                'vqemulti.method'],
      install_requires=['numpy', 'scipy', 'openfermion', 'posym', 'cirq', 'pennylane', 'qiskit']
      )
