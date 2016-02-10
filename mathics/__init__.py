#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from __future__ import absolute_import
import sys
import codecs
writer = codecs.getwriter("utf-8")
sys.stdout = writer(sys.stdout)

from mathics.version import __version__
from mathics.core.expression import (
    Expression, Symbol, String, Number, Integer, Real, Complex, Rational,
    from_python)
from mathics.core.convert import from_sympy


def get_version():
    version = {}

    import sympy
    import mpmath

    from django.core.exceptions import ImproperlyConfigured

    try:
        import django
        from django.conf import settings
        version['django'] = django.get_version()
    except (ImportError, ImproperlyConfigured):
        pass
    version['mathics'] = __version__
    version['sympy'] = sympy.__version__
    version['mpmath'] = mpmath.__version__
    version['python'] = sys.subversion[0] + " " + sys.version.split('\n')[0]
    return version


def get_version_string(is_server, newlines=False):
    version = get_version()
    result = []
    result.append(u"Mathics %s" % version['mathics'])
    result.append(u"on %s" % version['python'])
    libs = []
    if 'django' in version and is_server:
        libs.append("Django %s" % version['django'])
    libs += ["SymPy %s" % version['sympy'], "mpmath %s" % version['mpmath']]
    result.append(u"using %s" % ", ".join(libs))
    return ("\n" if newlines else " ").join(result)


def print_version(is_server):
    print("\n" + get_version_string(is_server, newlines=True))


def print_license():
    print(u"\n"
          u"Copyright (C) 2011-2015 The Mathics Team.\n"
          u"This program comes with ABSOLUTELY NO WARRANTY.\n"
          u"This is free software, and you are welcome to redistribute it\n"
          u"under certain conditions.\n"
          u"See the documentation for the full license.\n")

from mathics.core.parser import parse, ScanError, ParseError
