#!/usr/bin/env python

from distutils.core import setup
from glob import glob

scripts = glob('bin/*')

setup(name='wslurm',
      version='0.2',
      description='Colorful clusters',
      long_description=open('README.rst').read(),
      author='Tom Daff',
      author_email='tdd20@cam.ac.uk',
      license='BSD',
      packages=['wslurm'],
      scripts=scripts,
      classifiers=["Programming Language :: Python",
                   "Programming Language :: Python :: 3",
                   "Development Status :: 3 - Alpha",
                   "Intended Audience :: Science/Research",
                   "Intended Audience :: System Administrators",
                   "License :: OSI Approved :: BSD License",
                   "Operating System :: OS Independent",
                   "Topic :: System :: Monitoring"])
