#!/usr/bin/env python
# pylint: disable=invalid-name
"""set up mini-caffe"""
from __future__ import absolute_import
import os
from setuptools import setup


current_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = ['../build', '../build/Release']
dll_path = [os.path.join(current_dir, p) for p in dll_path]
if os.name == 'nt':
    dll_path = [os.path.join(p, 'caffe.dll') for p in dll_path]
else:
    dll_path = [os.path.join(p, 'libcaffe.so') for p in dll_path]
lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]


setup(name='minicaffe',
      version='0.4.0',
      packages=['minicaffe'],
      description=open(os.path.join(current_dir, 'README.md')).read(),
      install_requires=['numpy'],
      url='https://github.com/luoyetx/mini-caffe',
      data_files=[('minicaffe', [lib_path[0]])])
