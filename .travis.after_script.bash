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
#   Rene Milk       (2016, 2018)
#
# ~~~

if [[ $TRAVIS_JOB_NUMBER == *.1 ]] ; then
    git config --global hooks.clangformat ${CLANG_FORMAT}
    CHECK_DIR=${SUPERDIR}/${MY_MODULE}
    PYTHONPATH=${SUPERDIR}/scripts/python/ python3 -c "import travis_report as tp; tp.clang_format_status(\"${CHECK_DIR}\")"
    ${SRC_DCTRL} ${BLD} --only=${MY_MODULE} configure
    ${SRC_DCTRL} ${BLD} --only=${MY_MODULE} make doc
    if [ "X${TRAVIS_PULL_REQUEST}" != "Xfalse" ] ; then
        ${SUPERDIR}/.travis/init_sshkey.sh ${encrypted_95fb78800815_key} ${encrypted_95fb78800815_iv} keys/dune-community/dune-community.github.io
        ${SUPERDIR}/.travis/deploy_docs.sh ${MY_MODULE} "${DUNE_BUILD_DIR}"
    fi
fi
