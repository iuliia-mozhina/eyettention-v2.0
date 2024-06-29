#!/bin/bash

# This is the build script for the different SWIFT interfaces.
# When executed, this should generate the ./SIM directory and
# write to it the serial CLI version, the parallel CLI version,
# the MPI version and the Python module. Additionally, the Python
# module is installed in the ./MCMC directory.
# You may change the compiler call (export CC=gcc) or the MPI
# wrapper (export MPICC=mpicc) in order to change the presets.
# It is recommended that you use a GNU-compatible C compiler.
# clang (Apple’s default compiler) only supports the serial
# CLI version but no other interface! If $CC was not set by you,
# the script will try to find gcc-8 and then gcc. If $MPICC
# was not set, the script will try to find mpicc.

export SWPAR_FILE=swpar.cfg
export BUILD_DIR=./SIM
export PYINSTALL_DIR=./MCMC
export RINSTALL_DIR=./MCMC
export TEMP_DIR=$BUILD_DIR/tmp
export PARDEF_FILE=$TEMP_DIR/tmp_pardef.c
export CCOUT=$TEMP_DIR/build.log
export BASENAME=swiftstat7
export PYEXEC=python2.7
export REXEC=R

if [ ! $CC ]; then
	if which gcc-8 2>/dev/null >/dev/null ; then
		export CC=gcc-8
	else
		export CC=gcc
	fi
fi

export DO_MPI=0


#if [ ! $MPICC ]; then
#	if which mpicc 2>/dev/null >/dev/null ; then
#		export MPICC=mpicc
#		export DO_MPI=1
#	fi
#else
#	export DO_MPI=1
#fi

if [ $DO_MPI = 1 ]; then
	export CC="$(mpicc -show | awk "{\$1= \"$CC\"; print \$0}")"
fi

export LDSHARED="$CC -shared"

echo $CC

echo "Compile executables in $BUILD_DIR" >&2
echo "Build Python module in $BUILD_DIR/py/lib* and install in $PYINSTALL_DIR" >&2
echo "Build R module in $BUILD_DIR/r and install in $RINSTALL_DIR" >&2
echo "Write build log file to $CCOUT" >&2
echo "Use $CC for compiling C code" >&2

mkdir -p $BUILD_DIR 2>/dev/null
mkdir -p $TEMP_DIR 2>/dev/null

allgood=1

echo "--- BEGIN GENERATE PARAMETER FILE ---" > $CCOUT
# Generate parameter C code
echo "// Note: This is a temporary file and will be overwritten when the $0 script is called." > $PARDEF_FILE
echo "// It is recommended to edit $SWPAR_FILE instead, from which $PARDEF_FILE file is generated." >> $PARDEF_FILE
$PYEXEC generate_pardef.py $SWPAR_FILE 2>>$CCOUT >> $PARDEF_FILE
genpar=$?
echo "--- END GENERATE PARAMETER FILE ---" >> $CCOUT
if [ $genpar != 0 ]; then
	echo "ERROR generating temporary parameter definition file ($PARDEF_FILE)! See $CCOUT for details. Please fix that issue before compiling can take place." >&2
	exit 2
fi

echo "--- BEGIN BUILD PARALLEL CLI VERSION ---"  >> $CCOUT
$CC -O3 -fopenmp C-CODE/swiftstat7_cli.c -I $TEMP_DIR -o $BUILD_DIR/${BASENAME}p -lm $* 2>> $CCOUT 1>> $CCOUT
if [ $? != 0 ]; then
	echo "FAIL (1/5) building base parallel CLI version! See $CCOUT for details." >&2
	allgood=0
else
	echo "DONE (1/5) building base parallel CLI version ($BUILD_DIR/${BASENAME}p, $($BUILD_DIR/${BASENAME}p -v))" >&2
fi
echo "--- END BUILD PARALLEL CLI VERSION ---" >> $CCOUT


echo "--- BEGIN BUILD SERIAL (NON-PARALLEL) CLI VERSION ---"  >> $CCOUT
$CC -O3 -DDISABLE_THREADS C-CODE/swiftstat7_cli.c -I $TEMP_DIR -o $BUILD_DIR/$BASENAME -lm $* 2>> $CCOUT >> $CCOUT
if [ $? != 0 ]; then
	echo "FAIL (2/5) building base serial (non-parallel) CLI version! See $CCOUT for details." >&2
	allgood=0
else
	echo "DONE (2/5) building base serial (non-parallel) CLI version ($BUILD_DIR/$BASENAME, $($BUILD_DIR/$BASENAME -v))" >&2
fi
echo "--- END BUILD SERIAL (NON-PARALLEL) CLI VERSION ---" >> $CCOUT


echo "--- BEGIN BUILD PYTHON MODULE ---"  >> $CCOUT
export PY_CFLAGS="$*"
$PYEXEC pysetup.py build --force --build-base $BUILD_DIR/py --build-temp $BUILD_DIR/tmp 2>> $CCOUT >> $CCOUT && cp $BUILD_DIR/py/lib*/* $PYINSTALL_DIR 2>> $CCOUT >> $CCOUT
if [ $? != 0 ]; then
	echo "FAIL (3/5) building and installing Python module! See $CCOUT for details." >&2
	allgood=0
else
	echo "DONE (3/5) building and installing Python module ($(echo $BUILD_DIR/py/lib*/*), $PYINSTALL_DIR/$BASENAME.so, $(cd $PYINSTALL_DIR ; $PYEXEC -c "import $BASENAME" 2>&1))" >&2
fi
echo "--- END BUILD PYTHON MODULE ---" >> $CCOUT

echo "--- BEGIN BUILD R MODULE ---"  >> $CCOUT
export PKG_CFLAGS="-I $TEMP_DIR $*"
mkdir -p $BUILD_DIR/r 2>/dev/null
if [ $DO_MPI = 1 ]; then
	$REXEC CMD SHLIB -D SWIFT_MPI C-CODE/swiftstat7_r.c -o $BUILD_DIR/r/${BASENAME}_r.so --clean --preclean 2>> $CCOUT >> $CCOUT && cp $BUILD_DIR/r/* $RINSTALL_DIR 2>> $CCOUT >> $CCOUT
else
	$REXEC CMD SHLIB C-CODE/swiftstat7_r.c -o $BUILD_DIR/r/${BASENAME}_r.so --clean --preclean 2>> $CCOUT >> $CCOUT && cp $BUILD_DIR/r/* $RINSTALL_DIR 2>> $CCOUT >> $CCOUT
fi
if [ $? != 0 ]; then
	echo "FAIL (4/5) building and installing R module! See $CCOUT for details." >&2
	allgood=0
else
	echo "DONE (4/5) building and installing R module ($(echo $BUILD_DIR/r/*), $RINSTALL_DIR/${BASENAME}_r.so)" >&2
fi
echo "--- END BUILD R MODULE ---" >> $CCOUT

if [ $DO_MPI = 1 ] ; then
	echo "--- BEGIN BUILD MPI VERSION ---"  >> $CCOUT
	$CC -I $TEMP_DIR -fopenmp -O3 C-CODE/swiftstat7_mpi.c -o $BUILD_DIR/${BASENAME}mpi -D SWIFT_MPI_MAIN -lm $* 2>> $CCOUT >> $CCOUT
	if [ $? != 0 ]; then
		echo "FAIL (5/5) building MPI version! See $CCOUT for details." >&2
		allgood=0
	else
		echo "DONE (5/5) building MPI version ($BUILD_DIR/${BASENAME}mpi)" >&2
	fi
	echo "--- END BUILD MPI VERSION ---" >> $CCOUT
else
	echo "SKIP (5/5) building MPI version (no mpicc found)" >&2
fi

# Delete parameter C code
#rm -r $TEMP_DIR

echo "Note: $TEMP_DIR contains a few temporary files that you may want to check (incl. the build log file $CCOUT). However, none of its contents are crucial for running SWIFT."

if [ $allgood = 1 ]; then
	echo "All compilations successfully completed!" >&2
	exit 0
else
	echo "At least one compilation failed!" >&2
	exit 1
fi

