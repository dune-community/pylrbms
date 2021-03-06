#!/bin/bash
#
# ~~~
# This file is part of the pylrbms project:
#   https://github.com/dune-community/pylrbms
# Copyright 2010-2018 pylrbms developers and contributors. All rights reserved.
# License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
#          with "runtime exception" (http://www.dune-project.org/license.html)
# Authors:
#   Felix Schindler (2017)
#   Rene Milk       (2016 - 2018)
#
# ~~~

set -ex

WAIT="${SUPERDIR}/scripts/bash/travis_wait_new.bash 45"
source ${SUPERDIR}/scripts/bash/retry_command.bash

${SRC_DCTRL} ${BLD} --only=${MY_MODULE} configure
${SRC_DCTRL} ${BLD} --only=${MY_MODULE} make

# this does nothing if all current tests are distributed already, but triggers full build if not
# -> builder will timeout -> manually run refresh_test_timings -> push results
${SRC_DCTRL} ${BLD} --only=${MY_MODULE} bexec ninja -v -j 1 refresh_test_timings

free -h

if [ x"${TESTS}" == x ] ; then
    ${WAIT} ${SRC_DCTRL} ${BLD} --only=${MY_MODULE} bexec ninja -v test_binaries
    ${WAIT} ${SRC_DCTRL} ${BLD} --only=${MY_MODULE} bexec ctest -V -j 2
else
    ${WAIT} ${SRC_DCTRL} ${BLD} --only=${MY_MODULE} bexec ninja -v -j 1 test_binaries_builder_${TESTS}
    ${WAIT} ${SRC_DCTRL} ${BLD} --only=${MY_MODULE} bexec ctest -V -j 2 -L "^builder_${TESTS}$"
fi


# clang coverage currently disabled for being to mem hungry
if [[ ${CC} == *"clang"* ]] ; then
    exit 0
fi

pushd ${DUNE_BUILD_DIR}/${MY_MODULE}
COVERAGE_INFO=${PWD}/coverage.info
lcov --directory . --output-file ${COVERAGE_INFO} -c
for d in "dune-common" "dune-pybindxi" "dune-geometry"  "dune-istl"  "dune-grid" "dune-alugrid"  "dune-uggrid"  "dune-localfunctions" \
         "dune-xt-common" "dune-xt-functions" "dune-xt-la" "dune-xt-grid" ; do
    lcov --directory . --output-file ${COVERAGE_INFO} -r ${COVERAGE_INFO} "${SUPERDIR}/${d}/*"
done
lcov --directory . --output-file ${COVERAGE_INFO} -r ${COVERAGE_INFO} "${SUPERDIR}/${MY_MODULE}/dune/xt/*/test/*"
cd ${SUPERDIR}/${MY_MODULE}
${OLDPWD}/dune-env pip install codecov
${OLDPWD}/dune-env codecov -v -X gcov -X coveragepy -F ctest -f ${COVERAGE_INFO} -t ${CODECOV_TOKEN}
popd
