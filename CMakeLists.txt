cmake_minimum_required( VERSION 3.1 )
project(pylrbms CXX)

set( CMAKE_COLOR_MAKEFILE ON)

if(NOT (dune-common_DIR OR dune-common_ROOT OR
      "${CMAKE_PREFIX_PATH}" MATCHES ".*dune-common.*"))
    string(REPLACE  ${CMAKE_PROJECT_NAME} dune-common dune-common_DIR
      ${PROJECT_BINARY_DIR})
endif()

#find dune-common and set the module path
find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules"
  ${dune-common_MODULE_PATH})

#include the dune macros
include(DuneMacros)

# start a dune project with information from dune.module
dune_project()
dune_enable_all_packages()
add_subdirectory("cmake/modules")

include(DuneXTInstallPythonPackage)
# this symlinks all files in python/ to the binary dir and install into the virtualenv from there
# thereby making the compiled extensions directly usable from the venv
configure_file(python/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/python/setup.py)
dune_xt_install_python_package(PATH python)
include_dependent_binary_python_dirs()
# do not change order here
include(DuneUtils)
add_subdirectory(python)
add_subdirectory(dune/pylrbms/test)
dxt_add_make_dependent_bindings(dune-xt-common dune-xt-la dune-xt-functions dune-xt-grid dune-gdt)

add_pylicense()

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
