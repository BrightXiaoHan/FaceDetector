import numpy
from distutils.core import setup
from Cython.Build import cythonize


setup(
    name="torch_mtcnn",
    version="0.1",
    description='MTCNN pytorch implementation. Joint training and detecting together.',
    author='HanBing',
    author_email='beatmight@gmail.com',
    packages=['mtcnn', 'mtcnn.datasets', 'mtcnn.deploy', 'mtcnn.network', 'mtcnn.train', 'mtcnn.utils'],
    ext_modules=cythonize("mtcnn/utils/nms.pyx"),
    include_dirs=numpy.get_include()
)
