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
        Does the pattern match the tuple of expressions, exprs?
        Tracks named variables if they match.
        returns bool: True if the exprs match and False otherwise.
        '''
        raise NotImplementedError

    def unmatch(self, *args):
        '''
        Untracks named variables.
        returns None.
        '''
        pass


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
        if self.sub_patt.match(*exprs):
            count, known_exprs = self.names.get(self.name, (0, None))
            assert count >= 0
            if count > 0:
                # check equality with known_exprs
                if len(known_exprs) != len(exprs):
                    return False
                if not all(known_expr.same(expr) for known_expr, expr in zip(known_exprs, exprs)):
                    return False
            self.names[self.name] = (count + 1, exprs)
            return True
        return False

    def unmatch(self, *args):
        count, known_args = self.names[self.name]
        assert count >= 1
        if count == 1:
            del self.names[self.name]
        else:
            self.names[self.name] = (count - 1, args)
        self.sub_patt.unmatch(*args)


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
        self.match_index = None
        self.sub_patts = sub_patts

    def match(self, *exprs):
        for i, sub_patt in enumerate(self.sub_patts):
            if check_len_args(sub_patt, exprs) and sub_patt.match(*exprs):
                self.match_index = i
                return True
        return False

    def unmatch(self, *exprs):
        self.sub_patts[self.match_index].unmatch(*exprs)
        self.match_index = None


class ExpressionPattern(CompiledPattern):
    '''
    Represents a raw expression.
    '''
    min_args = 1
    max_args = 1

    def __init__(self, patt):
        self.expr = patt

    def match(self, expr):
        if self.expr.is_atom() and expr.is_atom():
            return self.expr.same(expr)
        elif not self.expr.is_atom() and not expr.is_atom():
            return match_expr(expr, self.expr)
        return False


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
        return ExpressionPattern(patt)


def check_len_args(slot, args):
    if slot.min_args > len(args):
        return False
    if slot.max_args is not None and len(args) > slot.max_args:
        return False
    return True


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

    # named patterns
    names = {}

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
        if check_len_args(slot, args) and slot.match(*args):
            yield [args]
            slot.unmatch(*args)
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
            slot.unmatch(*args[:i])


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
    ('f[1]', 'f[x_]', [0]),
    ('f[1, 2]', 'f[x_, x_]', None),
    ('f[1, 1]', 'f[x_, x_]', [0, 1]),
    ('f[1, 2, 1]', 'f[x_, y_, x_]', [0, 1, 2]),
    ('f[1, 2, 3, 1, 2]', 'f[x__, y__, x__]', [0, 0, 1, 2, 2]),
    ('f[1]', 'f[1|2]', [0]),
    ('f[3]', 'f[1|2]', None),
    ('f[1, 2]', 'f[x_Symbol|y_Integer, x_]', [0, 1]),
    ('f[a, 2]', 'f[x_Symbol|y_Integer, x_]', None),
    ('f[a, a]', 'f[x_Symbol|y_Integer, x_]', [0, 1]),
]

tests = [(_parse(expr), _parse(patt), result) for expr, patt, result in tests]


for expr, patt, result in tests:
    got = match_expr(expr, patt)
    if result != got:
        print('match_expr(%s, %s) = %s, expected %s' % (expr, patt, got, result))

stime = time()
for _ in range(1000):
    for expr, patt, result in tests:
        match_expr(expr, patt)
ftime = time()

print('duration: %.5f' % (ftime - stime))
