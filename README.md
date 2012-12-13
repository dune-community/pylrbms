```
# This file is part of the dune-gdt-pymor-interaction project:
#   https://github.com/dune-community/dune-gdt-pymor-interaction
# Copyright holders: Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```

[dune-gdt-pymor-interaction](https://github.com/dune-community/dune-gdt-pymor-interaction)
is a git supermodule which serves as a demonstration for the interaction between
[dune-gdt](https://github.com/dune-community/dune-gdt) and [pymor](http://pymor.org).


# Some notes on required software

* We recommend to use [docker](https://www.docker.com/) to ensure a fixed build environment.
  As a good starting point, take a look at our [Dockerfiles](https://github.com/dune-community/Dockerfiles) repository, which will guide you through the full process of working with docker and DUNE.
  While the compiled shared objects will (most likely) not work on your computer (they only work within the build environment of the container), you will have access to a jupyter notebook server from your computer.
* Compiler: we currently test gcc >= 4.9 and clang >= 3.8, other compilers may also work
* For a list of minimal (and optional) dependencies for several linux distributions, you can take a look our
  [Dockerfiles](https://github.com/dune-community/Dockerfiles) repository, e.g.,
  [debian/Dockerfile.minimal](https://github.com/dune-community/Dockerfiles/blob/master/debian/Dockerfile.minimal)
  for the minimal requirements on Debian jessie (and derived distributions).


# To build everything, do the following

First of all

## 1: checkout the repository and initialize all submodules:

```bash
mkdir -p $HOME/Projects/dune                 # <- adapt this to your needs
cd $HOME/Projects/dune
git clone https://github.com/dune-community/dune-gdt-pymor-interaction.git
cd dune-gdt-pymor-interaction
git submodule update --init --recursive
```

The next step depends on wether you are runnign in a specific docker container or directly on you machine.

## 2.a: Preparations within a docker container

Presuming you followed [these instructions](https://github.com/dune-community/Dockerfiles/blob/master/README.md) to get your docker setup working, and you just started and connected to a docker container by calling

```bash
./docker_run.sh debian-minimal-interactive dune-gdt-pymor-interaction /bin/bash
```

you are now left with an empty bash prompt (`exit` will get you out of there).
Issue the following commands:

```bash
export OPTS=gcc
cd $HOME/dune-gdt-pymor-interaction/debian-minimal #   <- this should match the docker container
source PATH.sh                                     #                             you are running
cd $BASEDIR
rm external-libraries.cfg ; ln -s debian-minimal/external-libraries.cfg . #         <- this also
```

Download and build all external libraries by calling (this _might_ take some time):

```bash
./local/bin/download_external_libraries.py
./local/bin/build_external_libraries.py
```

The next time you start the container you should at least issue the following commands before you start your work (you should also do this now to make use of the generated python virtualenv):

```bash
export OPTS=gcc
cd $HOME/dune-gdt-pymor-interaction/debian-minimal
source PATH.sh
```

## 2.b: Preparations on your machine

* Take a look at `config.opts/` and find settings and a compiler which suits your system, e.g. `config.opts/gcc`.
  The important part to look for is the definition of `CC` in these files: if, e.g., you wish to use clang in version 3.8 and clang is available on your system as `clang-3.8`, choose `OPTS=clang-3.8`; if it is available as `clang`, choose `OPTS=clang`.
  Select one of those options by defining
  
  ```bash
  export OPTS=gcc
  ```

  Note that dune-xt and dune-gdt do not build the Python bindings by default.
  You thus need to either

  - add `-DDUNE_XT_WITH_PYTHON_BINDINGS=TRUE` to the `CMAKE_FLAGS` of the selected config.opts file to set this permanently by calling
    ```bash
    echo "CMAKE_FLAGS=\"-DDUNE_XT_WITH_PYTHON_BINDINGS=TRUE "'${CMAKE_FLAGS}'"\"" >> config.opts/$OPTS
    ```

  - or
    ```
    export CMAKE_FLAGS="-DDUNE_XT_WITH_PYTHON_BINDINGS=TRUE ${CMAKE_FLAGS}"
    ```
    to set this temporarily,
  - or call `dunecontrol` twice (see below).
  
* Call

  ```bash
  ./local/bin/gen_path.py
  ```
  
  to generate a file `PATH.sh` which defines a local build environment. From now on you should source this file
  whenever you plan to work on this project, e.g. (depending on your shell):
  
  ```bash
  source PATH.sh
  ```

* Download and build all external libraries by calling (this _might_ take some time):

  ```bash
  ./local/bin/download_external_libraries.py
  ./local/bin/build_external_libraries.py
  ```

  This will in particular create a small Python virtualenv for the jupyter notebook, the configuration of which can be adapted by editing the virtualenv section in `external-libraries.cfg` (see below).
  This virtualenv will be activated from now on, whenever `PATH.sh` is sourced again (which you should do at this point):

  ```bash
  source PATH.sh
  ```
  If you do not wish to make use of the virtualenv, simply disable the respective section in `external-libraries.cfg`.

* To allow DUNE to find some of the locally built dependencies, you need to set the `CMAKE_INSTALL_PREFIX` by either

  - calling

    ```bash
    echo "CMAKE_FLAGS=\"-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} "'${CMAKE_FLAGS}'"\"" >> config.opts/$OPTS
    ```

    to set this permanently,
  
  - or by calling

    ```bash
    export CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} ${CMAKE_FLAGS}"
    ```
  
    to set this temporarily (recommended).

## 3: Build all DUNE modules

Using `cmake` and the selected options (this _will_ take some time):

```bash
./dune-common/bin/dunecontrol --opts=config.opts/$OPTS --builddir=$INSTALL_PREFIX/../build-$OPTS all
```
  
This creates a directory corresponding to the selected options (e.g. `build-gcc`) which contains a subfolder for each DUNE module.

If you did not add `-DDUNE_XT_WITH_PYTHON_BINDINGS=TRUE` to your `CMAKE_FLAGS` (see above), manually build the Python bindings by calling:

```bash
./dune-common/bin/dunecontrol --opts=config.opts/$OPTS --builddir=$INSTALL_PREFIX/../build-$OPTS bexec "make -j 1 bindings || echo no bindings"
```

## 4: Make use of the python bindings

The created Python bindings of each DUNE module are now available within the respective subdirectories of the build directory.
To make use of the bindings:

* Create and activate you favorite virtualenv with python3 as interpreter or use the prepared virtualenv:

  ```bash
  source PATH.sh
  ```

* Add the locations of interest to the Python interpreter of the virtualenv:

  ```bash
  for ii in dune-xt-common dune-xt-grid dune-xt-functions dune-xt-la dune-gdt; do echo "$INSTALL_PREFIX/../build-$OPTS/$ii" > "$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/$ii.pth"; done
  ```

* There is a bug in debian which might trigger an MPI init error when importing the Python modules (see for instance https://lists.debian.org/debian-science/2015/05/msg00054.html).
  As a workaround, set

  ```bash
  export OMPI_MCA_orte_rsh_agent=/bin/false
  ```

  or append this command to `PATH.sh` and source it again.

* There are jupyter notebooks available with some demos. Either `pip install notebook` in your favorite virtualenv or
  use the prepared one. Calling

  ```
  ./start_notebook_server.py
  ```

  should present you with an url which you can open in your favorite browser to show the notebooks.
