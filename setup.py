import setuptools

setuptools.setup(
  setup_requires=['pbr'],
  python_requires="~=3.6", # >= 3.6 < 4.0
  include_package_data=True,
  pbr=True
)

