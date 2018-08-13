// This file is part of the pylrbms project:
//   https://github.com/dune-community/pylrbms
// Copyright 2009-2018 pylrbms developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   Rene Milk       (2018)

#include "config.h"

#include <dune/pybindxi/pybind11.h>
#include <python/dune/xt/common/bindings.hh>


PYBIND11_MODULE(_bindings, m)
{
  namespace py = pybind11;

  Dune::XT::Common::bindings::add_initialization(m, "dune.pylrbms");
}
