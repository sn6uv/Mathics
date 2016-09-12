from mathics.core.expression import Integer, Symbol, Expression, strip_context


class Argument(object):
    def __init__(self, name):
        self.name = name


class IntegerArgument(Argument):
    pass


class RealArgument(Argument):
    pass


class Register(object):
    def __init__(self):
        self.value = None


class IntegerRegister(Register):
    pass


class RealRegister(Register):
    pass


class OpCode(object):
    pass

    def line(self):
        return '[???]'


class Return(OpCode):
    def __init__(self):
        pass

    def line(self):
        return 'Return'


class SetArgument(OpCode):
    def __init__(self, result, arg):
        self.result = result
        self.arg = arg

    def line(self):
        return 'I{} = A{}'.format(self.result.index, self.arg.index)


class SetConst(OpCode):
    def __init__(self, result, value):
        self.result = result
        self.value = value

    def line(self):
        return 'I{} = {}'.format(self.result.index, self.value)


class SetResult(OpCode):
    def __init__(self, result):
        self.result = result

    def line(self):
        return 'Result = I{}'.format(self.result.index)


class _RegularOpCode(OpCode):
    def __init__(self, result, *args, label=None):
        self.result = result
        self.args = args
        self.label = label

    @staticmethod
    def call(*args):
        return


class Add(_RegularOpCode):
    @staticmethod
    def call(arg1, arg2):
        return arg1 + arg2

    def line(self):
        return 'I{} = I{} + I{}'.format(self.result.index, self.args[0].index, self.args[1].index)


class Times(_RegularOpCode):
    @staticmethod
    def call(arg1, arg2):
        return arg1 * arg2

    def line(self):
        return 'I{} = I{} * I{}'.format(self.result.index, self.args[0].index, self.args[1].index)


class Power(_RegularOpCode):
    @staticmethod
    def call(arg1, arg2):
        return arg1 ** arg2

    def line(self):
        return 'I{} = I{} ^ I{}'.format(self.result.index, self.args[0].index, self.args[1].index)


class CastIntToReal(_RegularOpCode):
    @staticmethod
    def call(arg1):
        return float(arg1)

    def line(self):
        return 'R{} = I{}'.format(self.result.index, self.args[0].index)


class Compile(object):
    def __init__(self, args, expr):
        self.args = self.check_args(args)
        self.int_registers = []
        self.body = []
        return_reg = self.do_compile(expr)
        self.body.append(SetResult(return_reg))
        self.body.append(Return())
        self.enumerate_registers_and_args()

    def allocate_int_register(self):
        # returns new register
        reg = IntegerRegister()
        self.int_registers.append(reg)
        return reg

    def check_args(self, args):
        result = []
        for symb, t in args:
            if t == 'int':
                result.append(IntegerArgument(symb))
            else:
                # TODO Compile:ctyp2
                raise ValueError('Unknown argument type %s' % t)

        # check for duplicate arguments
        name_counts = {}
        for arg in result:
            name_counts[arg.name] = name_counts.get(arg.name, 0) + 1
        for name, count in name_counts.items():
            if count > 1:
                # TODO Compile:fdup
                raise ValueError('argument %s is repeated' % name)

        return result

    def do_compile(self, sub_expr):
        if isinstance(sub_expr, Integer):
            value = sub_expr.get_int_value()
            # TODO overflow check
            reg = self.allocate_int_register()
            self.body.append(SetConst(reg, value))
            return reg
        elif isinstance(sub_expr, Symbol):
            name = strip_context(sub_expr.get_name())
            for arg in self.args:
                if arg.name == name:
                    if isinstance(arg, IntegerArgument):
                        reg = self.allocate_int_register()
                        self.body.append(SetArgument(reg, arg))
                        return reg
            raise ValueError('Unknown symbol %s.' % name)
        elif isinstance(sub_expr, Expression):
            if sub_expr.has_form('Plus', 2):
                reg = self.allocate_int_register()
                left, right = sub_expr.get_leaves()
                left = self.do_compile(left)
                right = self.do_compile(right)
                self.body.append(Add(reg, left, right))
                return reg
            elif sub_expr.has_form('Times', 2):
                reg = self.allocate_int_register()
                left, right = sub_expr.get_leaves()
                left = self.do_compile(left)
                right = self.do_compile(right)
                self.body.append(Times(reg, left, right))
                return reg
            elif sub_expr.has_form('Power', 2):
                reg = self.allocate_int_register()
                left, right = sub_expr.get_leaves()
                left = self.do_compile(left)
                right = self.do_compile(right)
                self.body.append(Power(reg, left, right))
                return reg
            else:
                raise NotImplementedError()

    def debug_print(self):
        print('    %i arguments' % len(self.args))
        print('    %i integer registers' % len(self.int_registers))
        print()
        for i, op in enumerate(self.body):
            print('%2i %s' % (i, op.line()))

    def enumerate_registers_and_args(self):
        # TODO order registers by use order
        for i, reg in enumerate(self.int_registers):
            reg.index = i

        for i, arg in enumerate(self.args):
            arg.index = i

    def __call__(self, *args):
        # check number of args
        if len(args) != len(self.args):
            raise ValueError()

        # typecheck args
        for garg, earg in zip(args, self.args):
            if False:   # TODO
                raise ValueError()

        # initialise registers
        regs = [None for reg in self.int_registers]

        # register index of result
        result_index = None

        # run body
        for op in self.body:
            if isinstance(op, SetConst):
                regs[op.result.index] = op.value
            elif isinstance(op, SetArgument):
                regs[op.result.index] = args[op.arg.index]
            elif isinstance(op, SetResult):
                result_index = op.result.index
            elif isinstance(op, Return):
                break
            else:
                rargs = [regs[arg.index] for arg in op.args]
                regs[op.result.index] = op.call(*rargs)

        # return result
        return regs[result_index]


# Tests
cf = Compile([], Integer(1))
assert cf() == 1

cf = Compile([('x', 'int')], Integer(1))
assert cf(1) == 1

cf = Compile([('x', 'int')], Symbol('x'))
assert cf(5) == 5

cf = Compile([('x', 'int')], Expression('Plus', Symbol('x'), Integer(1)))
assert cf(5) == 6

cf = Compile([('x', 'int'), ('y', 'int')], Expression('Times', Integer(5), Expression('Plus', Symbol('x'), Symbol('y'))))
assert cf(2, 3) == 25

cf = Compile([('x', 'int')], Expression('Power', Symbol('x'), Integer(5)))
assert cf(2) == 32