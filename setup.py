#!/usr/bin/env python

from distutils.core import setup
from glob import glob

scripts = glob('bin/*')

setup(name='wsge',
      version='0.2',
      description='GridEngine scripts for the wooki/womble clusters',
      long_description=open('README.rst').read(),
      author='Tom Daff',
      author_email='tdd20@cam.ac.uk',
      license='BSD',
      url='http://bitbucket.org/tdaff/wsge/',
      packages=['wsge'],
      scripts=scripts,
      install_requires=['numpy', 'python_dateutil'],
      extras_requires={
          'lxml': 'lxml'},
      classifiers=["Programming Language :: Python",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3",
                   "Development Status :: 3 - Alpha",
                   "Intended Audience :: Science/Research",
                   "Intended Audience :: System Administrators",
                   "License :: OSI Approved :: BSD License",
                   "Operating System :: OS Independent",
                   "Topic :: System :: Monitoring"])
