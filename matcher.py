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


def match(expr, patt, evaluation):
    '''
    Checks if expr matches patt and returns a MatchObject
    returns None if the expression does not match.
    '''

    if patt.has_form({'Blank', 'BlankSequence', 'BlankNullSequence'}, None):
        n = len(patt.leaves)
        if n == 0:
            return MatchObject(expr)
        elif n == 1:
            if patt.leaves[0].same(expr.get_head()):
                return MatchObject(expr)
            else:
                return None
        else:
            return evaluation.message(patt.get_head_name(), 'argt', n)
    elif patt.has_form('Alternatives', None):
        for leaf in patt.leaves:
            m = match(expr, leaf, evaluation)
            if m is not None:
                return m
    elif patt.has_form({'Repeated', 'RepeatedNull'}, None):
        n = len(patt.leaves)
        valid = False

        if n == 1:
            valid = True
        elif n == 2:
            # second argument is count restrictions (range_spec)
            range_spec = patt.leaves[1]
            int_range_spec = range_spec.get_int_value()
            if int_range_spec is not None:
                # Repeated[_, 0] is allowed (MMA prohibits it).
                if int_range_spec is not None and int_range_spec < 0:
                    return evaluation.message(patt.get_head_name(), 'order', patt)
                elif int_range_spec >= 1:
                    valid = True
            elif range_spec.has_form('List', 1):
                int_range_spec = range_spec.leaves[0].get_int_value()
                # Repeated[_, {0}] is allowed.
                if int_range_spec is not None and int_range_spec < 0:
                    return evaluation.message(patt.get_head_name(), 'order', patt)
                elif int_range_spec is not None and int_range_spec == 1:
                    valid = True
            elif range_spec.has_form('List', 2):
                min_spec = range_spec.leaves[0].get_int_value()
                max_spec = range_spec.leaves[1].get_int_value()
                if min_spec is not None and max_spec is not None:
                    if min_spec < 0 or max_spec < 0:
                        return evaluation.message(patt.get_head_name(), 'range', patt)
                    elif min_spec > max_spec:
                        return evaluation.message(patt.get_head_name(), 'order', patt)
                    elif min_spec <= 1 <= max_spec:
                        valid = True
        if valid:
            return match(expr, patt.leaves[0], evaluation)
    elif patt.has_form('Pattern', None):
        n = len(patt.leaves)
        if n == 1:
            return evaluation.message('Pattern', 'argr', 'Pattern')
        elif n != 2:
            return evaluation.message('Pattern', 'argrx', 'Pattern', n)
        patt_name, sub_patt = patt.leaves
        name = patt_name.get_name()
        if not name:
            return evaluation.message('Pattern', 'patvar', patt)
        m = match(expr, sub_patt, evaluation)
        if m is not None:
            return MatchObject(expr, groups={name: m})
    elif patt.has_form('Except', None):
        n = len(patt.leaves)
        if n == 1:
            c = patt.leaves[0]
            p = Expression('Blank')
        elif n == 2:
            c, p = patt.leaves
        else:
            return evaluation.message('Except', 'argt', n)
        m_c = match(expr, c, evaluation)
        m_p = match(expr, p, evaluation)
        if m_c is None and m_p is not None:
            return MatchObject(expr)
    elif patt.has_form({'Longest', 'Shortest'}, None):
        # TODO
        raise NotImplementedError
    elif patt.has_form({'PatternSequence', 'OrderlessPatternSequence'}, None):
        n = len(patt.leaves)
        if n == 1:
            return match(expr, patt.leaves[0], evaluation)
    elif patt.has_form('Verbatim', None):
        n = len(patt.leaves)
        if n == 1:
            if patt.leaves[0].same(expr):
                return MatchObject(expr)
        else:
            return evaluation.message('Verbatim', 'argx', n)
    elif patt.has_form('HoldPattern', None):
        n = len(patt.leaves)
        if n == 1:
            return match(expr, patt.leaves[0], evaluation)
        else:
            return evaluation.message('HoldPattern', 'argx', n)
    elif patt.has_form('KeyValuePattern', None):
        # TODO
        raise NotImplementedError
    elif patt.has_form('Condition', None):
        n = len(patt.leaves)
        if n == 0:
            return evaluation.message('Condition', 'argm', n)
        elif n == 1:
            return evaluation.message('Condition', 'argmu')
        elif n == 2:
            sub_patt, test = patt.leaves
            m = match(expr, sub_patt, evaluation)
            if m is not None:
                named = {key: value.expr for key, value in m.groups.items()}
                test_expr = subs_named(named, test).evaluate(evaluation)
                if test_expr.is_true():
                    return m
    elif patt.has_form('PatternTest', None):
        n = len(patt.leaves)
        if n == 2:
            sub_patt, test = patt.leaves
            m = match(expr, sub_patt, evaluation)
            if m is not None:
                test_expr = Expression(test, expr).evaluate(evaluation)
                if test_expr.is_true():
                    return m
        elif n == 1:
            return evaluation.message('PatternTest', 'argr')
        else:
            return evaluation.message('PatternTest', 'argrx', n)
    elif patt.has_form('Optional', None):
        n = len(patt.leaves)
        if n in (1, 2):
            return match(expr, patt.leaves[0], evaluation)
        else:
            return evaluation.message('Optional', 'argt', n)
    else:
        # FIXME
        if patt.same(expr):
            return MatchObject(expr)
        else:
            return None


from mathics.core.parser import parse, SingleLineFeeder
from mathics.core.definitions import Definitions
from mathics.core.expression import Expression, Symbol
from mathics.core.evaluation import Evaluation, Output

print('loading builtins')
definitions = Definitions(add_builtin=True)
print('done')

tests = [
    ('a', '_', True),
    ('1', 'Alternatives[]', False),
    ('1', 'Alternatives[1]', True),
    ('1', 'Alternatives[2]', False),
    ('1', 'Alternatives[1, 2]', True),
    ('1', 'Alternatives[2, 1]', True),

    ('1', 'Repeated[1, 1]', True),
    ('1', 'Repeated[1, 0]', False), # MMA raises an error
    ('1', 'RepeatedNull[1, 0]', False),
    ('1', 'RepeatedNull[1, -1]', None),

    ('1', 'Repeated[1, {-1}]', None),
    ('1', 'RepeatedNull[1, {-1}]', None),
    ('1', 'Repeated[1, {0}]', False),
    ('1', 'RepeatedNull[1, {0}]', False),
    ('1', 'Repeated[1, {1}]', True),
    ('1', 'RepeatedNull[1, {1}]', True),

    ('1', 'Repeated[1, {0, 1}]', True),
    ('1', 'Repeated[1, {-1, 1}]', None),
    ('1', 'Repeated[1, {0, -1}]', None),
    ('1', 'Repeated[1, {0, 0}]', False),
    ('1', 'Repeated[1, {0, 1}]', True),
    ('1', 'Repeated[1, {1, 2}]', True),
    ('1', 'Repeated[1, {3, 3}]', False),
    ('1', 'Repeated[1, {2, 4}]', False),
    ('1', 'Repeated[1, {0, 4}]', True),

    ('1', 'Except[1]', False),
    ('1', 'Except[2]', True),
    ('1', 'Except[1, 2]', False),
    ('1', 'Except[2, 1]', True),
    ('1', 'Except[1, 1]', False),
    ('1', 'Except[2, 2]', False),
    ('1', 'Except[]', None),
    ('1', 'Except[1, 2, 3]', None),

    ('1', 'Verbatim[1]', True),
    ('x_', 'Verbatim[x_]', True),
    ('y_', 'Verbatim[x_]', False),
    ('1', 'Verbatim[]', None),
    ('1', 'Verbatim[1, 2]', None),

    ('1', 'HoldPattern[1]', True),
    ('1', 'HoldPattern[2]', False),
    ('1', 'HoldPattern[]', None),
    ('1', 'HoldPattern[1, 2]', None),

    # ('1', 'PatternSequence[]', False),
    # ('1', 'PatternSequence[1]', True),
    # ('1', 'PatternSequence[1, 2]', False),
    # ('1', 'OrderlessPatternSequence[]', False),
    # ('1', 'OrderlessPatternSequence[1]', True),
    # ('1', 'OrderlessPatternSequence[1, 2]', False),

    ('1', 'Condition[]', None),
    ('1', 'Condition[1]', None),
    ('1', 'Condition[1, 2, 3]', False),
    ('1', '1/;2', False),
    ('1', '1/;True', True),
    ('1', '2/;True', False),
    ('1', 'x_/;x>0', True),
    ('1', 'x_/;x>2', False),
    ('1', 'System`x_/;System`x>2', False),
    ('1', 'x_/;System`x>0', False),
    ('1', 'System`x_/;x>0', False),

    ('1', '_?NumberQ', True),
    ('1', '_?EvenQ', False),
    ('1', '_?fakeQ', False),
    ('1', 'PatternTest[]', None),
    ('1', 'PatternTest[1]', None),
    ('1', 'PatternTest[1, 2, 3]', None),

    ('1', 'Optional[]', None),
    ('1', 'Optional[1, 2, 3]', None),
    ('1', 'Optional[1]', True),
    ('1', 'Optional[2]', False),
    ('1', 'Optional[1, 0]', True),
    ('1', 'Optional[2, 0]', False),
]


def _parse(expr_string):
    return parse(definitions, SingleLineFeeder(expr_string))


class MessageException(Exception):
    def __init__(self, tag, name, *args):
        self.tag = tag
        self.name = name
        self.args = args


def message(tag, name, *args):
    raise MessageException(tag, name, *args)


class FakeOutput(Output):
    def max_stored_size(self, settings):
        return None


for expr, patt, result in tests:
    expr = _parse(expr)
    patt = _parse(patt)

    evaluation = Evaluation(definitions, catch_interrupt=False, output=FakeOutput())
    evaluation.message = message

    try:
        match_result = match(expr, patt, evaluation)
        match_result = bool(match_result)
    except MessageException as e:
        match_result = None

    if match_result is not result:
        print('MatchQ[%s, %s] returned %s expected %s' % (expr, patt, match_result, result))
        

