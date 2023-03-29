# coding: utf-8
from distutils.core import setup

setup(
    name='BEESEM',
    version='1.0.1',
    packages=['toolbox'],
    url='http://genetics.wustl.edu/gslab/resources/',
    license='GNU GPLv3',
    author='Shuxiang Ruan',
    author_email='sruan@wustl.edu',
    description='',
    scripts=['beesem.py', 'formatter.py'],
    py_modules=['SequenceModel', 'tools', 'universal'],
)
