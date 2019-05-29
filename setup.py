import contextlib
import os
import re
import sys
import platform
import subprocess
import glob

from codecs import open  # To use a consistent encoding
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
import distutils.log

# Get the current directory path.
dart_root = os.path.abspath(os.path.dirname(__file__))
dartpy_root = os.path.join(dart_root, 'python')

with open(os.path.join(dart_root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
description = 'Python API of Dynamic Animation and Robotics Toolkit.'

# distutils.log.set_verbosity(distutils.log.DEBUG)  # Set DEBUG level


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', sources=[]):
        Extension.__init__(self, name=name, sources=sources)
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.8.0':
                raise RuntimeError("CMake >= 3.8.0 is required on Windows")

        for ext in self.extensions:
            self.build_cmake(ext)
         
        super().run()

    def build_cmake(self, ext):
        cmake_args = [
            '-DDART_BUILD_DARTPY=ON', '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        self.announce("Preparing the build environment", level=3)

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j36']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Now that the necessary directories are created, build

        distutils.log.info("Configuring cmake project")
        self.announce("Configuring cmake project", level=distutils.log.INFO)

        # Change your cmake arguments below as necessary
        # Below is just an example set of arguments for building Blender as a Python module

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp,
                              env=env)

        self.announce("Building binaries", level=distutils.log.INFO)
        
        subprocess.check_call(['cmake', '--build', '.', '--target', 'dartpy'] +
                              build_args,
                              cwd=self.build_temp)
        # subprocess.check_call(['cmake', '--build', '.', '--target', 'install'],
        #                       cwd=self.build_temp)


# Set up the python package wrapping this extension.
setup(
    name='dartpy',
    version='0.0.1-10',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('dartpy')],
    url='https://github.com/dartsim/dart',
    author='Jeongseok Lee',
    author_email='jslee02@gmail.com',
    license='BSD 2-Clause',
    keywords='dartsim robotics',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Framework :: Robot Framework',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering',
    ],
    cmdclass=dict(build_ext=CMakeBuildExt, install_lib=install_lib),
)
