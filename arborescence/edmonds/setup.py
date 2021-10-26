from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='edmonds',
      ext_modules=[cpp_extension.CppExtension('edmonds', ['edmonds.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
