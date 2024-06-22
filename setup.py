import setuptools
import platform

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__


extra_compile_args = [ "-O3" ]
if platform.system() == "Windows":
  extra_compile_args = [ "/O2" ]

setuptools.setup(
  setup_requires=['pbr'],
  python_requires=">=3.8,<4.0",
  include_package_data=True,
  ext_modules=[
    setuptools.Extension(
      'mapbufferaccel',
      sources=[ 'mapbufferaccel.c' ],
      include_dirs=[ str(NumpyImport()) ],
      language='c',
      extra_compile_args=extra_compile_args,
    )
  ],
  pbr=True
)

