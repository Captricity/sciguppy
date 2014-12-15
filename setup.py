import os
from setuptools import setup
from setuptools.command.install import install

try:
    from post_setup import post_install
except ImportError:
    post_install = lambda: None

class CudaInstall(install):
    def run(self):
        install.run(self)
        post_install(os.path.join(self.install_lib, 'sciguppy'))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "sciguppy",
    version = "0.0.4",
    author="Captricity",
    author_email="support@captricity.com",
    description="SciGuppy is a library that accelerates scipy functions using the GPU",
    packages=["sciguppy"],
    package_data={'': ['**/*.cu']},
    cmdclass={
            'install': CudaInstall
        },
    scripts=['scripts/sciguppy_benchmark'],
    install_requires=read('requirements.txt')
)
