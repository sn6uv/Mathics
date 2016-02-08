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

#------------------------------------------------------------------------------
# Mathics runtime configuration
#------------------------------------------------------------------------------

# Recusion depth limit
MAX_RECURSION_DEPTH = 512

# Number of bits of precision for inexact calculations
MACHINE_PRECISION = 64

# Leave this True unless you have specific reason for not permitting users to
# access local files (like running mathics on a public webserver).
ENABLE_FILES_MODULE = True

# Time in seconds to allow for each execution. Set to `None` to disable.
TIMEOUT = None

# Set this True if you prefer 12 hour time to be the default
TIME_12HOUR = False

# Local time zone for this installation.
# On Windows set this to the same as your system time zone.
TIME_ZONE = 'Europe/Vienna'

#------------------------------------------------------------------------------
# System settings - don't edit unless you know what you are doing
#------------------------------------------------------------------------------

import pkg_resources
import sys
import os

# Root directory of mathics installation
ROOT_DIR = pkg_resources.resource_filename('mathics', '') + '/'
if sys.platform.startswith('win'):
    DATA_DIR = os.environ['APPDATA'].replace(os.sep, '/') + '/Python/Mathics/'
else:
    DATA_DIR = os.path.expanduser('~/.local/var/mathics/')

DOC_DIR = ROOT_DIR + 'doc/documentation/'
DOC_TEX_DATA = ROOT_DIR + 'doc/tex/data'
DOC_XML_DATA = ROOT_DIR + 'doc/xml/data'
DOC_LATEX_FILE = ROOT_DIR + 'doc/tex/documentation.tex'
