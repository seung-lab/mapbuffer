import setuptools

setuptools.setup(
  setup_requires=['pbr'],
  python_requires="~=3.6", # >= 3.6 < 4.0
  include_package_data=True,
  ext_modules=[
    setuptools.Extension(
      'mapbufferaccel',
      sources=[ 'mapbufferaccel.c' ],
      language='c',
      extra_compile_args=[ "-O3" ],
    )
  ],
  pbr=True
)

