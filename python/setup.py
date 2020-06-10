from setuptools import setup
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
	name = 'CUDAInline',
	version = '0.0.4',
	description = 'A CUDA interface for Python',
	long_description=long_description,
	long_description_content_type='text/markdown',  
	url='https://github.com/fynv/CUDAInline',
	license='Anti 996',
	author='Fei Yang',
	author_email='hyangfeih@gmail.com',
	keywords='GPU CUDA Python NVRTC',
	packages=['CUDAInline'],
	package_data = { 'CUDAInline': ['*.dll', '*.so']},
	install_requires = ['cffi','numpy'],	
)


