import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(  name = "cythoncode",
        cmdclass = {"build_ext": build_ext},
        ext_modules = [ Extension("mcts",
                                  sources=["mcts.pyx"],
                                  language='c++',   #using C++
                                  libraries=["m"],  #for using C's math lib
                                  extra_compile_args = ["-ffast-math"])],
        include_dirs=[numpy.get_include(),
                      os.path.join(numpy.get_include(), 'numpy')])
