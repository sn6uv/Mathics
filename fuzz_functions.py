from __future__ import print_function

import traceback


from mathics.core.expression import Expression, Integer, String, Symbol
from mathics.core.evaluation import Evaluation
from mathics.core.definitions import Definitions
from mathics.builtin import builtins

definitions = Definitions(add_builtin=True)

# arguments to test
all_args = [
    [],
    [Integer(0)],
    [Symbol('a')],
    [Integer(0), Symbol('a'), String('abc')]
]
    

for builtin_name in builtins:
    print('testing', builtin_name, end='')
    had_error = False
    for args in all_args:
        expr = Expression(builtin_name, *args)
        evaluation = Evaluation(definitions, catch_interrupt=False)
        try:
            evaluation.evaluate(expr, timeout=3)
        except KeyboardInterrupt:
            raise
        except:
            if not had_error:
                print()
            print(expr)
            had_error = True
            traceback.print_exc()
            print('-' * 70)
    if not had_error:
        print(' OK')
