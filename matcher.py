from __future__ import print_function


class PatternCompilationError(Exception):
    def __init__(self, tag, name, *args):
        self.tag = tag
        self.name = name
        self.args = args


class CompiledPattern(object):
    min_args = 1
    max_args = 1

    def match(self):
        raise NotImplementedError


class _BlankPattern(CompiledPattern):
    def __init__(self, patt):
        n = len(patt.leaves)
        if n == 0:
            self.expr = None
        elif n == 1:
            self.expr = patt.leaves[0]
        else:
            raise PatternCompilationError(patt.get_head_name(), 'argt', 'Blank', n, 0, 1)


class BlankPattern(_BlankPattern):
    min_args = 1
    max_args = 1

    def match(self, expr):
        return self.expr is None or self.expr.same(expr.get_head())


class BlankSequencePattern(_BlankPattern):
    min_args = 1
    max_args = None

    def match(self, *exprs):
        return self.expr is None or all(self.expr.same(expr.get_head()) for expr in exprs)


class BlankNullSequencePattern(_BlankPattern):
    min_args = 0
    max_args = None

    def match(self, *exprs):
        return self.expr is None or all(self.expr.same(expr.get_head()) for expr in exprs)


class PatternPattern(CompiledPattern):
    def __init__(self, patt):
        n = len(patt.leaves)
        if n == 2:
            name = patt[0].get_name()
            if not name:
                raise PatternCompilationError('Pattern', 'patvar', patt)
            sub_patt = compile_patt(patt)
            self.min_args = sub_patt.min_args
            self.max_args = sub_patt.max_args
            self.sub_patt = sub_patt
        raise PatternCompilationError('Pattern', 'argx', 'Pattern', n, 2)

    def match(self, *exprs):
        return self.sub_patt.match(self, *exprs)


class ExpressionPattern(CompiledPattern):
    '''
    Represents a raw expression.
    '''
    min_args = 1
    max_args = 1

    def __init__(self, patt):
        self.expr = patt

    def match(self, expr):
        return match_expr(expr, self.expr)


def compile_patt(patt):
    if patt.has_form('Blank', None):
        return BlankPattern(patt)
    elif patt.has_form('BlankSequence', None):
        return BlankSequencePattern(patt)
    elif patt.has_form('BlankNullSequence', None):
        return BlankNullSequencePattern(patt)
    elif patt.has_form('Pattern', None):
        return PatternPattern(patt)
    else:
        return ExpressionPattern(patt)


def match_expr(expr, patt):
    '''
    Matches the arguments of two expressions.

    e.g. Does f[1, 2] match f[_, _Integer]?

    returns a list of integers
    '''
    assert expr.get_head() is not None
    assert patt.get_head() is not None
    assert expr.get_head().same(patt.get_head())

    # empty case
    if not expr.leaves and not patt.leaves:
        return []

    head = expr.get_head()

    # TODO look at head attributes
    is_orderless = False
    is_flat = False

    # compile leaves
    slots = [compile_patt(leaf) for leaf in patt.leaves]

    # find the appropriate matching generator
    if not is_orderless and not is_flat:
        gen = gen_ordered_flatless(slots, expr.leaves)
    elif is_orderless and not is_flat:
        gen = gen_orderless_flat(slots, expr.leaves)
    elif not is_orderless and is_flat:
        gen = gen_ordered_flat(slots, expr.leaves)
    elif is_orderless and is_flat:
        gen = gen_orderless_flatless(slots, expr.leaves)

    # look for the first match
    try:
        match = next(gen)
    except StopIteration:
        return None

    # rewrite match as a list of slot indices
    assignments = []
    for j, slot in enumerate(match):
        for arg in slot:
            assignments.append(j)
    return assignments


def gen_ordered_flatless(slots, args):
    '''
    Ordered flatless argument assignment generator.
    '''
    if len(slots) == 1:
        slot = slots[0]
        if slot.min_args <= len(args) and (slot.max_args is None or len(args) <= slot.max_args) and slot.match(*args):
            yield [args]
    elif len(slots) > 1:
        slot = slots[0]
        start = slot.min_args
        end = (len(args) if slot.max_args is None else slot.max_args) + 1
        if start > len(args) + 1:
            return
        if end > len(args) + 1:
            end = len(args) + 1
        assert start <= end
        for i in range(start, end):
            if not slot.match(*args[:i]):
                continue
            for right in gen_ordered_flatless(slots[1:], args[i:]):
                yield [args[:i]] + right


def gen_orderless_flatless(slots, args):
    raise NotImplementedError


def gen_ordered_flat(slots, args):
    raise NotImplementedError


def gen_orderless_flat(slots, args):
    raise NotImplementedError


from time import time

from mathics.core.parser import parse, SingleLineFeeder
from mathics.core.definitions import Definitions
from mathics.core.expression import Expression, Symbol
from mathics.core.evaluation import Evaluation, Output


print('loading builtins')
definitions = Definitions(add_builtin=True)
print('done')


def _parse(expr_string):
    return parse(definitions, SingleLineFeeder(expr_string))


tests = [
    ('f[]', 'f[]', []),
    ('f[]', 'f[_]', None),
    ('f[1]', 'f[]', None),
    # ('f[1]', 'f[1]', [0]),
    ('f[1]', 'f[_]', [0]),
    ('f[1, 2]', 'f[_]', None),
    ('f[1, 2]', 'f[_, _]', [0, 1]),
    ('f[1, 2]', 'f[__, _]', [0, 1]),
    ('f[1, 2]', 'f[_, __]', [0, 1]),
    ('f[1, 2]', 'f[__, __]', [0, 1]),
    ('f[1, 2]', 'f[_, _, _]', None),
    ('f[1, 2, 3]', 'f[__, __]', [0, 1, 1]),
    ('f[1, 2, 3]', 'f[___, __]', [1, 1, 1]),
    ('f[1, 2, 3]', 'f[__, ___]', [0, 1, 1]),
    ('f[1, 2, 3]', 'f[___, ___]', [1, 1, 1]),
    ('f[1, 2, 3]', 'f[__, _, __]', [0, 1, 2]),
    ('f[1]', 'f[_Integer]', [0]),
    ('f[1]', 'f[_String]', None),
    ('f[a, b, 1, c]', 'f[__, _Integer, __]', [0, 0, 1, 2]),
    ('f[1, b, 1, c]', 'f[___, _Integer, _Integer, ___]', None),
    ('f[1]', 'f[_, ___]', [0]),
    ('f[1]', 'f[_, ___, _]', None),
]

tests = [(_parse(expr), _parse(patt), result) for expr, patt, result in tests]


for expr, patt, result in tests:
    got = match_expr(expr, patt)
    if result != got:
        print('match_expr(%s, %s) = %s, expected %s' % (expr, patt, got, result))

stime = time()
for _ in range(10000):
    for expr, patt, result in tests:
        match_expr(expr, patt)
ftime = time()

print('duration: %.5f' % (ftime - stime))
