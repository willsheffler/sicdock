#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'rpxdock'
DESCRIPTION = 'Protein Docking Toolkit'
URL = 'https://github.com/willsheffler/rpxdock.git'
EMAIL = 'willsheffler@gmail.com'
AUTHOR = 'Will Sheffler'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
   'numpy==1.16.3',
   'xarray==0.11.3',
   'pandas==0.24.2',
   'pytest',
   'pytest-repeat',
   'tqdm',
   'homog',
   'cppimport',
   'breathe',
   'cffi',
   'tabulate',
   'sphinx>3',
   'breathe',
   'exhale',
   'sphinx_rtd_theme>=0.3.1',
]

# What packages are optional?
EXTRAS = {
   'pyrosetta': 'pyrosetta',
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))
os.environ["CC"] = "gcc-7"

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
   with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
      long_description = '\n' + f.read()
except FileNotFoundError:
   long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
   project_slug = NAME.lower().replace('-', '_').replace(' ', '_')
   with open(os.path.join(here, project_slug, '__version__.py')) as f:
      exec(f.read(), about)
else:
   about['__version__'] = VERSION

class UploadCommand(Command):
   'Support setup.py upload.'

   description = 'Build and publish the package.'
   user_options = []

   @staticmethod
   def status(s):
      'Prints things in bold.'
      print('\033[1m{0}\033[0m'.format(s))

   def initialize_options(self):
      pass

   def finalize_options(self):
      pass

   def run(self):
      try:
         self.status('Removing previous builds…')
         rmtree(os.path.join(here, 'dist'))
      except OSError:
         pass

      self.status('Building Source and Wheel (universal) distribution…')
      os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

      self.status('Uploading the package to PyPI via Twine…')
      os.system('twine upload dist/*')

      self.status('Pushing git tags…')
      os.system('git tag v{0}'.format(about['__version__']))
      os.system('git push --tags')

      sys.exit()

def get_files(*patterns):
   fnames = set()
   from pathlib import Path
   for pattern in patterns:
      for filename in Path('rpxdock').glob(pattern):
         fnames.add(str(filename).lstrip('rpxdock/'))
   return list(fnames)

# Where the magic happens:
setup(
   name=NAME,
   version=about['__version__'],
   description=DESCRIPTION,
   long_description=long_description,
   long_description_content_type='text/rst',
   author=AUTHOR,
   author_email=EMAIL,
   python_requires=REQUIRES_PYTHON,
   url=URL,
   packages=find_packages(),
   # If your package is a single module, use this instead of 'packages':
   # py_modules=['rpxdock'],

   # entry_points={
   #     'console_scripts': ['mycli=mymodule:cli'],
   # },
   install_requires=REQUIRED,
   extras_require=EXTRAS,
   tests_require=['pytest'],
   package_dir={'rpxdock': 'rpxdock'},
   package_data=dict(rpxdock=[
      '*/*.hpp',
      '*/*.cpp',
      'data/*.pickle',
      'data/hscore/*',
      'data/pdb/*',
      'sampling/data/karney/*',
      'rotamer/richardson.csv',
   ] + get_files('extern/**/*')),
   include_package_data=True,
   license='Apache2',
   classifiers=[
      # Trove classifiers
      # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
      'License :: OSI Approved :: Apache Software License',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: Implementation :: CPython',
      'Programming Language :: Python :: Implementation :: PyPy'
   ],
   # $ setup.py publish support.
   cmdclass={
      'upload': UploadCommand,
   },
)
