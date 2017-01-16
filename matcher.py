from __future__ import print_function


class PatternCompilationError(Exception):
    def __init__(self, tag, name, *args):
        self.tag = tag
        self.name = name
        self.args = args


class CompiledPattern(object):
    min_args = 1
    max_args = 1

    def match(self, *exprs):
        '''
        Generates None as the pattern matches the tuple of expressions, exprs.
        Tracks named variables as they match.
        returns bool: True if the exprs match and False otherwise.
        '''
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
        if self.expr is None or self.expr.same(expr.get_head()):
            yield None


class BlankSequencePattern(_BlankPattern):
    min_args = 1
    max_args = None

    def match(self, *exprs):
        if self.expr is None or all(self.expr.same(expr.get_head()) for expr in exprs):
            yield None


class BlankNullSequencePattern(_BlankPattern):
    min_args = 0
    max_args = None

    def match(self, *exprs):
        if self.expr is None or all(self.expr.same(expr.get_head()) for expr in exprs):
            yield None


class PatternPattern(CompiledPattern):

    def __init__(self, patt, names):
        n = len(patt.leaves)
        if n == 2:
            name = patt.leaves[0].get_name()
            if not name:
                raise PatternCompilationError('Pattern', 'patvar', patt)
            sub_patt = compile_patt(patt.leaves[1], names)
            self.min_args = sub_patt.min_args
            self.max_args = sub_patt.max_args
            self.sub_patt = sub_patt
            self.name = name
            self.names = names
        else:
            raise PatternCompilationError('Pattern', 'argx', 'Pattern', n, 2)

    def match(self, *exprs):
        for _ in self.sub_patt.match(*exprs):
            count, known_exprs = self.names.get(self.name, (0, None))
            assert count >= 0
            if count > 0:
                # check equality with known_exprs
                if len(known_exprs) != len(exprs):
                    continue
                if not all(known_expr.same(expr) for known_expr, expr in zip(known_exprs, exprs)):
                    continue
            self.names[self.name] = (count + 1, exprs)
            yield None
            self.names[self.name] = (count, exprs)


class AlternativesPattern(CompiledPattern):
    '''
    Note that Alternatives behaves weirdly for named patterns. Names in unbound
    sub-patterns are used for replacement rules but not for checking the match.

    Example:

    >> f[g[x_]|y_Integer, x_] := {{x}, {y}}
    >> f[1, 2]      (* x not checked *)
     = {{2}, {1}}
    >> f[g[1], 2]   (* x bound and checked *)
     = f[g[1], 2]

    This behaviour is acheived by implementing in the naive way.
    '''

    def __init__(self, patt, names):
        sub_patts = [compile_patt(leaf, names) for leaf in patt.leaves]
        self.min_args = min(sub_patt.min_args for sub_patt in sub_patts)
        if any(sub_patt.max_args is None for sub_patt in sub_patts):
            self.max_args = None
        else:
            self.max_args = max(sub_patt.max_args for sub_patt in sub_patts)
        self.sub_patts = sub_patts

    def match(self, *exprs):
        for i, sub_patt in enumerate(self.sub_patts):
            if sub_patt.min_args <= len(exprs) and (sub_patt.max_args is None or len(exprs) <= sub_patt.max_args):
                for _ in sub_patt.match(*exprs):
                    yield None


class ExpressionPattern(CompiledPattern):
    '''
    Represents a raw expression.
    '''
    min_args = 1
    max_args = 1

    def __init__(self, patt, names):
        self.expr = patt
        self.names = names

    def match(self, expr):
        if self.expr.is_atom() and expr.is_atom():
            if self.expr.same(expr):
                yield None
        elif not self.expr.is_atom() and not expr.is_atom():
            old_names = self.names.copy()
            for _ in match_expr(expr, self.expr, self.names):
                yield None
                self.names.clear()
                self.names.update(old_names)


def compile_patt(patt, names):
    if patt.has_form('Blank', None):
        return BlankPattern(patt)
    elif patt.has_form('BlankSequence', None):
        return BlankSequencePattern(patt)
    elif patt.has_form('BlankNullSequence', None):
        return BlankNullSequencePattern(patt)
    elif patt.has_form('Pattern', None):
        return PatternPattern(patt, names)
    elif patt.has_form('Alternatives', None):
        return AlternativesPattern(patt, names)
    else:
        return ExpressionPattern(patt, names)


def match_expr(expr, patt, names=None):
    '''
    Matches the arguments of two expressions.

    e.g. Does f[1, 2] match f[_, _Integer]?

    :expr Expression: expression to match.
    :patt Expression: pattern against which to match.
    :names dict: predefined named match variables.
    :yields (list, dict): match positions for each argument and named matches.
    '''
    assert expr.get_head() is not None
    assert patt.get_head() is not None
    assert expr.get_head().same(patt.get_head())

    if names is None:
        names = {}

    # empty case
    if not expr.leaves and not patt.leaves:
        yield [], names
        return

    head = expr.get_head()

    # TODO look at head attributes
    is_orderless = False
    is_flat = False

    # compile leaves
    slots = [compile_patt(leaf, names) for leaf in patt.leaves]

    # find the appropriate matching generator
    if not is_orderless and not is_flat:
        gen = gen_ordered_flatless(slots, expr.leaves)
    elif is_orderless and not is_flat:
        gen = gen_orderless_flat(slots, expr.leaves)
    elif not is_orderless and is_flat:
        gen = gen_ordered_flat(slots, expr.leaves)
    elif is_orderless and is_flat:
        gen = gen_orderless_flatless(slots, expr.leaves)

    # look for matches
    for match in gen:
        # rewrite match as a list of slot indices
        assignments = []
        for j, slot in enumerate(match):
            for arg in slot:
                assignments.append(j)
        yield assignments, names
    return


def gen_ordered_flatless(slots, args):
    '''
    Ordered flatless argument assignment generator.
    '''
    if len(slots) == 1:
        slot = slots[0]
        if slot.min_args <= len(args) and (slot.max_args is None or len(args) <= slot.max_args):
            for _ in slot.match(*args):
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
            for _ in slot.match(*args[:i]):
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


_tests = [
    ('f[]', 'f[]', []),
    ('f[]', 'f[_]', None),
    ('f[1]', 'f[]', None),
    ('f[1]', 'f[1]', [0]),
    ('f[f[1]]', 'f[f[1]]', [0]),
    ('f[1]', 'f[f[1]]', None),
    ('f[f[1]]', 'f[1]', None),
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
    ('f[1]', 'f[x_]', [0], {'x': (1, '1')}),
    ('f[1, 2]', 'f[x_, x_]', None),
    ('f[1, 1]', 'f[x_, x_]', [0, 1], {'x': (2, '1')}),
    ('f[1, 2, 1]', 'f[x_, y_, x_]', [0, 1, 2], {'x': (2, '1'), 'y': (1, '2')}),
    ('f[1, 2, 3, 1, 2]', 'f[x__, y__, x__]', [0, 0, 1, 2, 2], {'x': (2, '1', '2'), 'y': (1, '3')}),
    ('f[1]', 'f[1|2]', [0]),
    ('f[3]', 'f[1|2]', None),
    ('f[1, 2]', 'f[x_Symbol|y_Integer, x_]', [0, 1], {'y': (1, '1'), 'x': (1, '2')}),
    ('f[a, 2]', 'f[x_Symbol|y_Integer, x_]', None),
    ('f[a, a]', 'f[x_Symbol|y_Integer, x_]', [0, 1], {'x': (2, 'a')}),
    ('f[f[1]]', 'f[f[x_]]', [0], {'x': (1, '1')}),
    ('f[f[1], 1]', 'f[f[x_], x_]', [0, 1], {'x': (2, '1')}),
    ('f[f[1], 2]', 'f[f[x_], x_]', None),
    ('f[f[1], 2]', 'f[f[x_]|y_, x_]', [0, 1], {'x': (1, '2'), 'y': (1, 'f[1]')}),
]


# parse tests
tests = []
for test in _tests:
    if len(test) == 3:
        expr, patt, slots = test
        names = {}
    elif len(test) == 4:
        expr, patt, slots, names = test
    names = {_parse(key).get_name(): (values[0], tuple(_parse(value) for value in values[1:])) for key, values in names.items()}
    tests.append((_parse(expr), _parse(patt), (slots, names)))


for expr, patt, result in tests:
    try:
        got = next(match_expr(expr, patt))
    except StopIteration:
        got = None, {}

    if result != got:
        print('match_expr(%s, %s) = %s, expected %s' % (expr, patt, got, result))

stime = time()
for _ in range(1000):
    for expr, patt, result in tests:
        for _ in match_expr(expr, patt):
            pass
ftime = time()

print('duration: %.5f' % (ftime - stime))
