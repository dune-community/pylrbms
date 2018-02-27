#!/usr/bin/env python3
#
# This file is part of the dune-xt-common project:
#   https://github.com/dune-community/dune-xt-common
# Copyright 2009-2018 dune-xt-common developers and contributors. All rights reserved.
# License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Felix Schindler (2017)
#   Rene Milk       (2016 - 2018)
#
#      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
#          with "runtime exception" (http://www.dune-project.org/license.html)

import sys
import os
from setuptools import setup, find_packages

setup(name='pylrbms',
      version='0.1',
      namespace_packages=['dune'],
      description='LRBMS with pyMOR and dune-gdt',
      author='',
      author_email='',
      url='',
      packages = find_packages(),
      zip_safe = 0,
      package_data = {'': ['*.so']},
      install_requires=['pymor'],
      dependency_links = [
        'git+https://zivgitlab.uni-muenster.de:{token}@zivgitlab.uni-muenster.de/srave_01/pymor.git@master#egg=pymor-0.5'
        .format(token=os.environ.get('ZIVGITLAB_TOKEN', 'git'))
        ],
      scripts=[])
