from setuptools import setup, find_packages
import numpy


def readme():
    with open('README.rst') as f:
        return f.read()


INSTALL_REQUIRES = ['numpy', 'scipy']

EXTRAS_REQUIRE = {'tests': ['pytest', 'pytest-cov'],
                  'docs': ['sphinx', 'sphinx-gallery',
                           'sphinx_rtd_theme', 'numpydoc',
                           'matplotlib', 'download']
                  }

if __name__ == "__main__":
    setup(packages=find_packages(),
          include_dirs=[numpy.get_include()],
          install_requires=INSTALL_REQUIRES,
          extras_require=EXTRAS_REQUIRE,
          version='0.7.5',
          description='Simulation of Low Density Parity Check Codes ldpc',
          long_description=readme(),
          classifiers=[
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Telecommunications Industry',
          'Natural Language :: English',
          ],
          keywords='codes ldpc error detection decoding coding pyldpc',
          url='https://github.com/hichamjanati/pyldpc',
          author='Hicham Janati',
          author_email='hicham.janati@inria.fr',
          license='MIT',
          )
