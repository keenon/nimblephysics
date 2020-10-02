import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', target=None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.target = target


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        if ext.target is not None:
            build_args += ['--target', ext.target]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            # We need this on the manylinux2010 Docker images to find the correct Python
            if platform.system() == 'Linux':
                cmake_args += ['-DPYTHON_INCLUDE_DIR=/opt/python/cp38-cp38/include/python3.8/']
                cmake_args += ['-DPYTHON_LIBRARY=/opt/python/cp38-cp38/lib/python3.8/']
            build_args += ['--', '-j14']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp, env=env)
        # Create the __init__.py in the library folder, so that delocate-wheel works properly
        Path(extdir+"/__init__.py").touch()


setup(
    name='diffdart',
    version='0.0.2',
    author='Keenon Werling',
    author_email='keenonwerling@gmail.com',
    description='A differentiable fully featured physics engine',
    long_description='',
    license='MIT',
    package_dir={'': 'python'},
    packages=['diffdart'],
    ext_package='diffdart_libs',
    ext_modules=[CMakeExtension('cmake_example', target='_diffdart')],
    install_requires=[
        'torch',
        'numpy'
    ],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
