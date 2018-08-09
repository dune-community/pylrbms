# ~~~
# This file is part of the pylrbms project:
#   https://github.com/dune-community/pylrbms
# Copyright 2009-2018 pylrbms developers and contributors. All rights reserved.
# License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Rene Milk (2018)
# ~~~

import pytest
from dune.xt.common.test import load_all_submodule

def test_load_all():
    import dune.pylrbms as dpy
    load_all_submodule(dpy)


