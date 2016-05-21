from setuptools import setup

setup(name='ghgforcing',
      version='0.1',
      description='Calculate radiative forcing from GHG emissions',
      url='https://github.com/gschivley/ghgforcing',
      author='Greg Schivley',
      license='MIT',
      packages=['ghgforcing'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'random'],
      zip_safe=False)