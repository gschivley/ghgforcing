from setuptools import setup

setup(name='ghgforcing',
      version='0.1.3',
      description='Calculate radiative forcing from GHG emissions',
      url='https://github.com/gschivley/ghgforcing',
      author='Greg Schivley',
      author_email='greg.schivley@gmail.com',
      license='MIT',
      packages=['ghgforcing'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas'],
      zip_safe=False)