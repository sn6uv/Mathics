# -*- coding: utf8 -*-

u"""
    Mathics: a general-purpose computer algebra system
    Copyright (C) 2011-2013 The Mathics Team

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pkg_resources
import sys
import os
from os import path


TIMEOUT = None
MAX_RECURSION_DEPTH = 512

# number of bits of precision for inexact calculations
MACHINE_PRECISION = 64

ROOT_DIR = pkg_resources.resource_filename('mathics', '') + '/'
if sys.platform.startswith('win'):
    DATA_DIR = os.environ['APPDATA'].replace(os.sep, '/') + '/Python/Mathics/'
else:
    DATA_DIR = path.expanduser('~/.local/var/mathics/')

DOC_DIR = ROOT_DIR + 'doc/documentation/'
DOC_TEX_DATA = ROOT_DIR + 'doc/tex/data'
DOC_XML_DATA = ROOT_DIR + 'doc/xml/data'
DOC_LATEX_FILE = ROOT_DIR + 'doc/tex/documentation.tex'

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'Europe/Vienna'

# Set this True if you prefer 12 hour time to be the default
TIME_12HOUR = False

# Leave this True unless you have specific reason for not permitting
# users to access local files
ENABLE_FILES_MODULE = True
