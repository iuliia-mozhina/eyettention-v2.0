
# This script is used to build and install the Python interface
# (swiftstat7_py.c) for SWIFT. Note that macOS may need static
# linking in order to enable OpenMP multithreading. When called
# from build.sh, the relevant directories are read from their
# respective environment variables.


from numpy.distutils.core import setup, Extension
import os, platform

more_args = []


if os.getenv('PY_CFLAGS',''):
	more_args = os.environ['PY_CFLAGS'].split(" ")

if os.getenv('DO_MPI', '0') == '1':
	more_args.append('-DSWIFT_MPI')


if platform.system() == "Darwin":
	# use static linking to make OpenMP work on Mac
	modules = [Extension(os.getenv('BASENAME','swift'), sources = ['C-CODE/swiftstat7_py.c'], include_dirs=[os.getenv('TEMP_DIR','./SIM/tmp')], extra_compile_args=['-fopenmp']+more_args, extra_link_args=['-fopenmp','-lpython2.7','-static']+more_args)]
else:
	modules = [Extension(os.getenv('BASENAME','swift'), sources = ['C-CODE/swiftstat7_py.c'], include_dirs=[os.getenv('TEMP_DIR','./SIM/tmp')], extra_compile_args=['-fopenmp']+more_args, extra_link_args=['-fopenmp','-lpython2.7']+more_args)]
#module1 = Extension('swift', sources = ['swiftstat7_py.c'])


setup (name = 'swiftstat',
       description = 'Python module interface to SWIFT API',
       ext_modules = modules)

