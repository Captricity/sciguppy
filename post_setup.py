"""
Post install script for pyCUDA applications to warm the cubin cache
"""

import os
import glob
import pycuda.autoinit
from pycuda.compiler import SourceModule

def post_install(install_path):
    CACHE_DIR = os.path.join(install_path, 'cache')
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    for kernel in glob.glob(os.path.join(install_path, 'kernel', '*.cu')):
        SourceModule(open(kernel).read(), cache_dir=CACHE_DIR)
