from __future__ import print_function


class MatchObject(object):
    def __init__(self, expr, groups=None):
        self.expr = expr

        if groups is None:
            self.groups = {}
        else:
            self.groups = groups

    def __repr__(self):
        return '<MatchObject(expr=%s, groups=%s)>' % (self.expr, self.groups)


def subs_named(named, expr):
    '''
    Substitutes named patterns in expr.
    '''
    if expr.is_symbol():
        return named.get(expr.get_name(), expr)
    elif expr.is_atom():
        return expr
    else:
        new_head = subs_named(named, expr.get_head())
        new_leaves = [subs_named(named, leaf) for leaf in expr.leaves]
        return Expression(new_head, *new_leaves)


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
    else:
        return ExpressionPattern(patt)


def sum_capacity(patts):
    '''
    returns min and max capacity from a list of compiled patterns.
    max capacity can be None.
    '''
    if not patts:
        return 0, 0
    assert not any(patt.min_args is None for patt in patts)
    min_cap = sum(patt.min_args for patt in patts)
    if any(patt.max_args is None for patt in patts):
        max_cap = None
    else:
        max_cap = sum(patt.max_args for patt in patts)
    assert max_cap is None or min_cap <= max_cap
    return min_cap, max_cap


def match_expr(expr, patt):
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

    # find the matching generator
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
        if (slot.min_args <= len(args) and
            (slot.max_args is None or len(args) <= slot.max_args) and
            slot.match(*args)):

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


for expr, patt, result in tests:
    expr = _parse(expr)
    patt = _parse(patt)

    got = match_expr(expr, patt)
    if result != got:
        print('match_expr(%s, %s) = %s, expected %s' % (expr, patt, got, result))
