from setuptools import setup,find_packages


def readme():
	with open('README.rst') as f:
		return f.read()

if __name__ == "__main__":

	setup(name='pyldpc',
		version='0.7.5',
		description='Simulation of Low Density Parity Check Codes ldpc',
		long_description=readme(),
		classifiers=[
			'Programming Language :: Python :: 2.7',
			'Programming Language :: Python :: 3.5',
			'Development Status :: 4 - Beta',
			'Environment :: MacOS X',
			'Framework :: IPython',
			'Intended Audience :: Developers',
			'Intended Audience :: Education',
			'Intended Audience :: Science/Research',
			'Intended Audience :: Telecommunications Industry',
			'Natural Language :: English',
			'Natural Language :: French'
		],
		keywords='codes ldpc error detection decoding coding pyldpc',
		url='https://github.com/hichamjanati/pyldpc',
		author='Hicham Janati',
		author_email='hicham.janati@ensae.fr',
		license='MIT',
		packages = ['pyldpc'],
		install_requires = ['numpy','scipy'],
		zip_safe=False)
