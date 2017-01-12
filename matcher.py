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
    min_capacity = None
    max_capacity = None
    is_ordered = True

    def match_single(self):
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

    def match_single(self, expr):
        return self.expr is None or self.expr.same(expr.get_head())


class BlankPattern(_BlankPattern):
    min_capacity = 1
    max_capacity = 1


class BlankSequencePattern(_BlankPattern):
    min_capacity = 1
    max_capacity = None


class BlankNullSequencePattern(_BlankPattern):
    min_capacity = 0
    max_capacity = None


class ExpressionPattern(CompiledPattern):
    '''
    Represents a raw expression.
    '''

    def __init__(self, patt):
        self.expr = patt

    def match_single(self, expr):
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
    assert not any(patt.min_capacity is None for patt in patts)
    min_cap = sum(patt.min_capacity for patt in patts)
    if any(patt.max_capacity is None for patt in patts):
        max_cap = None
    else:
        max_cap = sum(patt.max_capacity for patt in patts)
    assert max_cap is None or min_cap <= max_cap
    return min_cap, max_cap


def match_expr(expr, patt):
    # FIXME
    # if expr.get_head() is None or patt.get_head() is None:
    #     raise ValueError
    # if not expr.get_head().same(patt.get_head()):
    #     raise ValueError

    head = expr.get_head()

    # TODO look at head attributes
    is_orderless = False
    is_flat = False

    # compile leaves
    slots = [compile_patt(leaf) for leaf in patt.leaves]

    # # fast exit by patt capacity
    # min_cap, max_cap = sum_capacity(slots)
    # if min_cap > len(expr.leaves) or (max_cap is not None and max_cap < len(expr.leaves)):
    #     return None

    # TODO greedy matching

    if not is_orderless and not is_flat:
        assignments = solve(slots, expr.leaves)
    elif is_orderless and not is_flat:
        assignments = solve_orderless(slots, expr.leaves)
    elif not is_orderless and is_flat:
        # assignments = solve_flat(slots, expr.leaves)
        pass
    elif is_orderless and is_flat:
        # assignments = solve_orderless_flat(slots, expr.leaves)
        pass
    return assignments


def solve(slots, args):
    '''
    backtracking solver O(n^2) worst case but expected O(n) for e.g. f[__, __, ...]
    '''
    assignments = []    # assigment[arg_index] = slot_index
    i = 0   # slot index
    j = 0   # how many args bound to current slot
    backtrack = False    # did we arrive at this state by backtracking
    while len(assignments) < len(args):
        # print(assignments)
        if not backtrack and i < len(slots) and j >= slots[i].min_capacity:

            # move to next slot
            backtrack = False
            i += 1
            j = 0
        elif (i < len(slots) and
            (slots[i].max_capacity is None or j < slots[i].max_capacity) and
            slots[i].match_single(args[len(assignments)])):

            # assign arg to slot i
            backtrack = False
            assignments.append(i)
            j += 1
        else:
            # backtrack to previous slot
            backtrack = True

            if i == 0:
                # no solution
                return None
            while assignments and assignments[-1] == i:
                assignments.pop()
            i -= 1

            # count how many args assigned to new slot
            j = 0
            while j < len(assignments) and assignments[-(j+1)] == i:
                j += 1

    # all args are assigned sucessfully
    assert sorted(assignments) == assignments
    assert len(assignments) == len(args)
    assert all(slots[si].match_single(args[ai]) for ai, si in enumerate(assignments))

    # check remaining slots are satisfied
    while i < len(slots):
        if slots[i].min_capacity > j:
            return None
        j = 0
        i += 1

    # all slots satisfied
    for i, slot in enumerate(slots):
        n = sum(1 for assignment in assignments if assignment == i)
        assert slot.min_capacity <= n
        assert slot.max_capacity is None or slot.max_capacity >= n

    return assignments


def solve_orderless(slots, args):
    # TODO Fulkerson-Ford solver
    pass


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
]


for expr, patt, result in tests:
    expr = _parse(expr)
    patt = _parse(patt)

    got = match_expr(expr, patt)
    if result != got:
        print('match_expr(%s, %s) = %s, expected %s' % (expr, patt, got, result))
