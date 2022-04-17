from setuptools import setup

exec(open('causalnet/version.py').read())

setup(name = 'causalnet',
      version = __version__,
      description = 'A Python package to streamline the process of creating embeddings out of your categorical data & build causal inference models using the Dragonnet architecture',
      author = 'DJ',
      author_email = 'willofdeepak@gmail.com',
      license = '',
      packages = ['causalnet'],
      python_requires = '>=3.6',
      install_requires = ['pandas>=0.24.1', 
                          'sklearn',
                          'scipy',
                          'matplotlib',
                          'numpy~=1.19.2',
                          'tensorflow~=2.5.0']
    )