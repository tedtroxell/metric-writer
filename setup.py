#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(name='metric_writer',
      version='0.0.1',
      description='Simple and easy metric logger for PyTorch',
      author='Ted Troxell',
      author_email='ted@tedtroxell.com',
      url='https://tedtroxell.com',
      zip_safe=False,
      packages=find_packages(),
     )