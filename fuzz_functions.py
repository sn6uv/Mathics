from __future__ import print_function

import sys
import traceback
import itertools


from mathics.core.expression import (
    Expression, Integer, String, Symbol, Real, Rational)
from mathics.core.evaluation import Evaluation
from mathics.core.definitions import Definitions
from mathics.builtin import builtins

definitions = Definitions(add_builtin=True)
builtin_names = sorted(builtins.keys())

sep = 120 * '-'


def build_args(max_len=2):
    'arguments to test.'
    single_args = [
        Integer(0), Symbol('a'), String("abc"), Real(0.5), Rational(1, 2),
        Symbol('Null'), Expression('DirectectedInfinity', Integer(1)),
        Symbol('Indeterminate'),
        Expression('Plus', Symbol('a'), Symbol('b')),
        Expression('List', Symbol('a'), Symbol('b'), Symbol('c')),
    ]

    all_args = itertools.chain.from_iterable(
        itertools.permutations(single_args, r) for r in range(max_len + 1))
    return list(all_args)


def test_builtin(builtin_name, all_args):
    print('testing', builtin_name, end='... ')
    sys.stdout.flush()
    had_error = False
    stop = False
    for args in all_args:
        expr = Expression(builtin_name, *args)
        evaluation = Evaluation(definitions, catch_interrupt=False)
        try:
            evaluation.evaluate(expr, timeout=5)
        except KeyboardInterrupt:
            stop = True
            break
        except:
            if not had_error:
                print('FAIL')
                print(sep)
            print(expr)
            had_error = True
            traceback.print_exc()
            print(sep)
    if not had_error:
        print('OK')
    return had_error, stop


def main():
    failed_builtins = []
    all_args = build_args()
    for builtin_name in builtin_names:
        had_error, stop = test_builtin(builtin_name, all_args)
        if had_error:
            failed_builtins.append(builtin_name)
        if stop:
            break
    if failed_builtins:
        print('Failed builtins:')
        print('\n'.join('- ' + name for name in failed_builtins))


if __name__ == '__main__':
    main()
