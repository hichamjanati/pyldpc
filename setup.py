from setuptools import setup, find_packages
import numpy as np


def readme():
    with open('README.rst') as f:
        return f.read()


INSTALL_REQUIRES = ['numpy', 'scipy', 'numba']

EXTRAS_REQUIRE = {'tests': ['pytest', 'pytest-cov'],
                  'docs': ['sphinx', 'sphinx-gallery',
                           'sphinx_rtd_theme', 'numpydoc',
                           'matplotlib', 'download']
                  }

if __name__ == "__main__":
    setup(name="pyldpc",
          packages=find_packages(),
          include_dirs=[np.get_include()],
          extras_require=EXTRAS_REQUIRE,
          version='0.7.8',
          description='Simulation of Low Density Parity Check (LDPC) Codes',
          install_requires=INSTALL_REQUIRES,
          long_description=readme(),
          classifiers=[
              'Programming Language :: Python :: 3.6',
              'Development Status :: 4 - Beta',
              'Intended Audience :: Developers',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Telecommunications Industry',
              'Natural Language :: English',
          ],
          keywords='codes ldpc error decoding information theory',
          url='https://github.com/hichamjanati/pyldpc',
          author='Hicham Janati',
          author_email='hicham.janati100@gmail.com',
          license='MIT',
          )
