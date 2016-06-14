#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals

import six

from mathics.core.parser.parser import Parser
from mathics.core.parser.convert import convert
from mathics.core.parser.feed import SingleLineFeeder, MultiLineFeeder
from mathics.core.expression import ensure_context


parser = Parser()


def parse_convert(definitions, feeder):
    ast = parser.parse(feeder)
    if ast is not None:
        return convert(ast, definitions)
    else:
        return None


def parse(definitions, feeder):
    '''
    Parse input (from the frontend, -e, input files, ToExpression etc).
    Look up symbols according to the Definitions instance supplied.

    Feeder must implement the feed and empty methods, see core/parser/feed.py.
    '''
    return parse_convert(definitions, feeder)


class SystemDefinitions(object):
    """
    Dummy Definitions object that puts every unqualified symbol in
    System`.
    """
    def lookup_name(self, name):
        assert isinstance(name, six.string_types)
        return ensure_context(name)


def parse_builtin_rule(string):
    '''
    Parse rules specified in builtin docstrings/attributes. Every symbol
    in the input is created in the System` context.
    '''
    return parse_convert(SystemDefinitions(), SingleLineFeeder(string))


def parse_code(code, definitions):
    return parse(definitions, SingleLineFeeder(code))


def parse_lines(lines, definitions, yield_lineno=False):
    '''
    Given a list of lines of code yield expressions until all code is parsed.

    If yield_lines is True return `(lineno, expr)` otherwise just `expr`.

    The line number (lineno) corresponds to the start of the expression.

    A generator is used so that each expression can be evaluated before
    continuing; the parser is dependent on defintions and evaluation may change
    the definitions.
    '''
    if isinstance(lines, six.text_type):
        lines = lines.splitlines()

    feeder = MultiLineFeeder(lines)

    while not feeder.empty():
        lineno = feeder.lineno
        expression = parse_convert(definitions, feeder)
        if expression is not None:
            if yield_lineno:
                yield (expression, lineno)
            else:
                yield expression
