import sys
import sympy
import mpmath

from mathics.version import __version__
from mathics.core.expression import (
    Expression, Symbol, String, Number, Integer, Real, Complex, Rational,
    from_python)
from mathics.core.convert import from_sympy

version_info = {
    'mathics': __version__,
    'sympy': sympy.__version__,
    'mpmath': mpmath.__version__,
    'python': sys.subversion[0] + " " + sys.version.split('\n')[0],
}

version_string = 'Mathics {mathics}\non {python}\nusing SymPy {sympy}, mpmath {mpmath}'.format(**version_info)

license_string = """
Copyright (C) 2011-2016 The Mathics Team.
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.
See the documentation for the full license.
"""

# this import needs to be last due to a circular dependency on version_string
from mathics.core.parser import parse, ScanError, ParseError
