import sympy
import sympy.stats
from mathics.builtin.base import SympyFunction, Builtin
from mathics.core.expression import Symbol, Complex, Integer
from mathics.core.convert import from_sympy, sympy_symbol_prefix


class DistributionParam(object):
    msg = ''

    @staticmethod
    def convert(value):
        return


class RealParam(DistributionParam):
    msg = 'realprm'

    @staticmethod
    def convert(value):
        if isinstance(value, Complex):
            return None
        else:
            # TODO
            return value.to_sympy()


class PositiveRealParam(DistributionParam):
    msg = 'posprm'

    @staticmethod
    def convert(value):
        if isinstance(value, Symbol):
            return sympy.Symbol(sympy_symbol_prefix + value.name, positive=True)
        elif isinstance(value, Complex):
            return None
        else:
            # TODO
            return value.to_sympy()


class IntegerParam(DistributionParam):
    msg = 'intprm'

    @staticmethod
    def convert(value):
        if isinstance(value, Symbol):
            # TODO
            return value.to_sympy()
        elif isinstance(value, Integer):
            return value.to_sympy()
        else:
            return None


class PositiveIntParam(DistributionParam):
    msg = 'pintprm'

    @staticmethod
    def convert(value):
        if isinstance(value, Integer):
            result = value.to_sympy()
            if result > 0:
                return result
        return None


class _SympyDistribution(SympyFunction):
    params = ()      # sequence of DistributionParam
    sympy_name = None

    def to_sympy(self, expr, **kwargs):
        # convert params
        if len(expr.leaves) != len(self.params):
            return
        params = [param.convert(leaf) for param, leaf in zip(self.params, expr.leaves)]
        if None in params:
            return

        # convert distribution
        try:
            return getattr(sympy.stats, self.sympy_name)('x', *params)
        except ValueError as e:
            return


class CDF(Builtin):
    '''
    >> CDF[NormalDistribution[mu, sigma]]
     = 1 / 2 + Erf[Sqrt[2] (-mu + #1) / (2 sigma)] / 2&
    '''

    rules = {
        'CDF[dist_, x_]': 'CDF[dist][x]',
    }

    def apply(self, dist, evaluation):
        'CDF[dist_]'
        dist = dist.to_sympy()
        try:
            result = sympy.stats.cdf(dist)
        except ValueError:
            return
        return from_sympy(result.simplify())


class PDF(Builtin):
    '''
    >> PDF[NormalDistribution[0, 1], x]
     = Sqrt[2] E ^ (-x ^ 2 / 2) / (2 Sqrt[Pi])
    '''

    rules = {
        'PDF[dist_, x_]': 'PDF[dist][x]',
    }

    def apply(self, dist, evaluation):
        'PDF[dist_]'
        dist = dist.to_sympy()
        dummy_arg = sympy.Symbol('PDFDummyArg')
        try:
            result = sympy.stats.density(dist)
            result = sympy.Lambda(dummy_arg, result.pdf(dummy_arg))
        except ValueError:
            return
        return from_sympy(result.simplify())


class InverseCDF(Builtin):
    '''
    >> InverseCDF[NormalDistribution[0, 1]]
     = Sqrt[2] InverseErfc[2 - 2 #1]&
    '''

    def apply(self, dist, evaluation):
        'InverseCDF[dist_]'
        dist = dist.to_sympy()
        try:
            result = sympy.stats.density(dist)
            result = result._inverse_cdf_expression()
        except ValueError:
            return
        return from_sympy(result.simplify())


class NormalDistribution(_SympyDistribution):
    '''
    >> CDF[NormalDistribution[mu, sigma]]
     = 1 / 2 + Erf[Sqrt[2] (-mu + #1) / (2 sigma)] / 2&

    >> CDF[NormalDistribution[0, 1]]
     = 1 - Erfc[Sqrt[2] #1 / 2] / 2&
    '''

    params = (RealParam, PositiveRealParam)
    sympy_name = 'Normal'

    rules = {
        'NormalDistribution[]': 'NormalDistribution[0, 1]',
    }
