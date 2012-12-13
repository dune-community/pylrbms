export BASEDIR=/home/falbr_01/dune-gdt-pymor-interaction
export INSTALL_PREFIX=$BASEDIR/manylinux-full/local
export PATH=$INSTALL_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib64:$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$INSTALL_PREFIX/lib64/pkgconfig:$INSTALL_PREFIX/lib/pkgconfig:$INSTALL_PREFIX/share/pkgconfig:$PKG_CONFIG_PATH
source /opt/openmpi-1.4.activate.sh
source /opt/gcc-4.9-toolchain.activate.sh
source /opt/intel-tbb-4.2_20140601.activate.sh
source /opt/cmake-3.7.2.activate.sh
source /opt/python/cp36-cp36m.activate.sh
export PATH=$HOME/.local/bin:$PATH
export CC=gcc
export CXX=g++
export F77=gfortran
export CMAKE_FLAGS="-DMETIS_ROOT=$INSTALL_PREFIX -DBOOST_ROOT=$INSTALL_PREFIX -DEIGEN3_INCLUDE_DIR=$INSTALL_PREFIX/include/eigen3 -DDUNE_XT_WITH_PYTHON_BINDINGS=TRUE -DTBB_INCLUDE_DIR=/opt/intel-tbb-4.2_20140601/usr/include -DTBB_LIBRARY_DIR=/opt/intel-tbb-4.2_20140601/usr/lib"
[ -e $INSTALL_PREFIX/bin/activate ] && . $INSTALL_PREFIX/bin/activate
export OMP_NUM_THREADS=1
