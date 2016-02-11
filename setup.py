#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from six.moves import range

"""Setuptools based setup script for Mathics.

For the easiest installation just type the following command (you'll probably
need root privileges):

    python setup.py install

This will install the library in the default location. For instructions on
how to customize the install procedure read the output of:

    python setup.py --help install

In addition, there are some other commands:

    python setup.py clean -> will clean all trash (*.pyc and stuff)

To get a full list of avaiable commands, read the output of:

    python setup.py --help-commands

Or, if all else fails, feel free to write to the sympy list at
mathics-users@googlegroups.com and ask for help.
"""

import sys
import platform
from setuptools import setup, Command, Extension

# Ensure user has the correct Python version
if sys.version_info[:2] != (2, 7) and sys.version_info < (3, 2):
    print("Mathics does not support Python %d.%d" % sys.version_info[:2])
    sys.exit(-1)

# stores __version__ in the current namespace
exec(compile(open('mathics/version.py').read(), 'mathics/version.py', 'exec'))

is_PyPy = (platform.python_implementation() == 'PyPy')

try:
    if is_PyPy:
        raise ImportError
    from Cython.Distutils import build_ext
except ImportError:
    EXTENSIONS = []
    CMDCLASS = {}
    INSTALL_REQUIRES = []
else:
    EXTENSIONS = {
        'core': ['expression', 'numbers', 'rules', 'pattern'],
        'builtin': ['arithmetic', 'numeric', 'patterns', 'graphics']
    }
    EXTENSIONS = [
        Extension('mathics.%s.%s' % (parent, module),
                  ['mathics/%s/%s.py' % (parent, module)])
        for parent, modules in EXTENSIONS.items() for module in modules]
    CMDCLASS = {'build_ext': build_ext}
    INSTALL_REQUIRES = ['cython>=0.15.1']

# General Requirements
INSTALL_REQUIRES += ['sympy==0.7.6', 'django >= 1.8, < 1.9', 'ply>=3.8',
                     'mpmath>=0.19', 'python-dateutil', 'colorama', 'six']


def subdirs(root, file='*.*', depth=10):
    for k in range(depth):
        yield root + '*/' * k + file


class initialize(Command):
    """
    Manually creates the database used by Django
    """

    description = "manually create the database used by django"
    user_options = []  # distutils complains if this is not here.

    def __init__(self, *args):
        self.args = args[0]  # so we can pass it to other classes
        Command.__init__(self, *args)

    def initialize_options(self):  # distutils wants this
        pass

    def finalize_options(self):    # this too
        pass

    def run(self):
        import os
        import subprocess
        settings = {}
        exec(compile(open('mathics/settings.py').read(), 'mathics/settings.py', 'exec'), settings)

        database_file = settings['DATABASES']['default']['NAME']
        print("Creating data directory %s" % settings['DATA_DIR'])
        if not os.path.exists(settings['DATA_DIR']):
            os.makedirs(settings['DATA_DIR'])
        print("Creating database %s" % database_file)
        try:
            subprocess.check_call(
                [sys.executable, 'mathics/manage.py', 'migrate', '--noinput'])
            print("")
            print("database created successfully.")
        except subprocess.CalledProcessError:
            print("error: failed to create database")
            sys.exit(1)


class test(Command):
    """
    Runs the unittests
    """

    description = "runs the unittests"
    user_options = []

    def __init__(self, *args):
        self.args = args[0]  # so we can pass it to other classes
        Command.__init__(self, *args)

    def initialize_options(self):  # distutils wants this
        pass

    def finalize_options(self):    # this too
        pass

    def run(self):
        import unittest
        test_loader = unittest.defaultTestLoader
        test_runner = unittest.TextTestRunner(verbosity=3)
        test_suite = test_loader.discover('test/')
        test_result = test_runner.run(test_suite)

        if not test_result.wasSuccessful():
            sys.exit(1)


CMDCLASS['initialize'] = initialize
CMDCLASS['test'] = test

mathjax_files = list(subdirs('media/js/mathjax/'))

setup(
    name="Mathics",
    cmdclass=CMDCLASS,
    ext_modules=EXTENSIONS,
    version=__version__,

    packages=[
        'mathics',
        'mathics.core',
        'mathics.builtin', 'mathics.builtin.pymimesniffer', 'mathics.data',
        'mathics.doc',
        'mathics.autoload',
        'mathics.packages',
        'mathics.web', 'mathics.web.templatetags'
    ],

    install_requires=INSTALL_REQUIRES,

    package_data={
        'mathics.doc': ['documentation/*.mdoc', 'xml/data'],
        'mathics.web': [
            'media/css/*.css', 'media/img/*.gif',
            'media/js/innerdom/*.js', 'media/js/prototype/*.js',
            'media/js/scriptaculous/*.js', 'media/js/three/Three.js',
            'media/js/three/Detector.js', 'media/js/*.js', 'templates/*.html',
            'templates/doc/*.html'] + mathjax_files,
        'mathics.data': ['*.csv', 'ExampleData/*'],
        'mathics.builtin.pymimesniffer': ['mimetypes.xml'],
        'mathics.autoload': ['formats/*/Import.m', 'formats/*/Export.m'],
        'mathics.packages': ['*/*.m', '*/Kernel/init.m'],
    },

    entry_points={
        'console_scripts': [
            'mathics = mathics.main:main',
            'mathicsserver = mathics.server:main',
        ],
    },

    # don't pack Mathics in egg because of sqlite database, media files, etc.
    zip_safe=False,

    # metadata for upload to PyPI
    author="Jan Poeschko",
    author_email="jan@poeschko.com",
    description="A general-purpose computer algebra system.",
    license="GPL",
    keywords="computer algebra system mathics mathematica sage sympy",
    url="http://www.mathics.org/",   # project home page, if any

    # TODO: could also include long_description, download_url, classifiers,
    # etc.
)
