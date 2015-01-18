import os
import glob
from setuptools import setup
from setuptools.command.install import install

def post_install(install_path):
    """
    Post install script for pyCUDA applications to warm the cubin cache
    """
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    CACHE_DIR = os.path.join(install_path, 'cache')
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    for kernel in glob.glob(os.path.join(install_path, 'kernel', '*.cu')):
        SourceModule(open(kernel).read(), cache_dir=CACHE_DIR)

class CudaInstall(install):
    def run(self):
        install.run(self)
        post_install(os.path.join(self.install_lib, 'sciguppy'))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

all_requirements = read('requirements.txt').split()
setup_requirements = filter(lambda r: 'pycuda' in r or 'pytools' in r, all_requirements)
install_requirements = filter(lambda r: not r.startswith('git'), all_requirements)
dependency_links = filter(lambda r: r.startswith('git'), all_requirements)

setup(
    name = "sciguppy",
    version = "0.0.7",
    author="Captricity",
    author_email="support@captricity.com",
    description="SciGuppy is a library that accelerates scipy functions using the GPU",
    packages=["sciguppy"],
    package_data={'': ['**/*.cu']},
    cmdclass={
            'install': CudaInstall
        },
    scripts=['scripts/sciguppy_benchmark'],
    setup_requires=setup_requirements,
    install_requires=install_requirements,
    dependency_links=dependency_links
)
