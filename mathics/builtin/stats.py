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
    <dl>
    <dt>'CDF[$dist$, $x$]'
      <dd>returns the cumulative distribution for the distribution $dist$ at $x$.
    <dt>'CDF[$dist$, {$x_1$, $x_2$, ...}]'
      <dd>returns the CDF evaluated at '{$x_1$, $x_2$, ...}'.
    <dt>'CDF[$dist$]'
      <dd>returns the CDF as a pure function.
    </dl>

    >> CDF[NormalDistribution[mu, sigma], x]
     = 1 / 2 + Erf[Sqrt[2] (-mu + x) / (2 sigma)] / 2

    >> CDF[NormalDistribution[mu, sigma], {x, y}]
     = {1 / 2 + Erf[Sqrt[2] (-mu + x) / (2 sigma)] / 2, 1 / 2 + Erf[Sqrt[2] (-mu + y) / (2 sigma)] / 2}

    >> CDF[NormalDistribution[mu, sigma]]
     = 1 / 2 + Erf[Sqrt[2] (-mu + #1) / (2 sigma)] / 2&
    '''

    rules = {
        'CDF[dist_, x_]': 'CDF[dist][x]',
        'CDF[dist_, xs_List]': 'CDF[dist] /@ xs',
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
    <dl>
    <dt>'PDF[$dist$, $x$]'
      <dd>returns the probability density function for the distribution $dist$ at $x$.
    <dt>'PDF[$dist$, {$x_1$, $x_2$, ...}]'
      <dd>returns the PDF evaluated at '{$x_1$, $x_2$, ...}'.
    <dt>'PDF[$dist$]'
      <dd>returns the PDF as a pure function.
    </dl>

    >> PDF[NormalDistribution[0, 1], x]
     = Sqrt[2] E ^ (-x ^ 2 / 2) / (2 Sqrt[Pi])

    >> PDF[NormalDistribution[0, 1]]
     = Sqrt[2] Exp[-#1 ^ 2 / 2] / (2 Sqrt[Pi])&

    >> PDF[NormalDistribution[0, 1], {x, y}]
     = {Sqrt[2] E ^ (-x ^ 2 / 2) / (2 Sqrt[Pi]), Sqrt[2] E ^ (-y ^ 2 / 2) / (2 Sqrt[Pi])}
    '''

    rules = {
        'PDF[dist_, x_]': 'PDF[dist][x]',
        'PDF[dist_, xs_List]': 'PDF[dist] /@ xs',
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
    <dl>
    <dt>'InverseCDF[$dist$, $q$]'
      <dd>returns the inverse cumulative distribution for the distribution $dist$ as a function of $q$.
    <dt>'InverseCDF[$dist$, {$x_1$, $x_2$, ...}]'
      <dd>returns the inverse CDF evaluated at '{$x_1$, $x_2$, ...}'.
    <dt>'InverseCDF[$dist$]'
      <dd>returns the inverse CDF as a pure function.
    </dl>

    >> InverseCDF[NormalDistribution[0, 1], x]
     = Sqrt[2] InverseErfc[2 - 2 x]

    >> InverseCDF[NormalDistribution[0, 1], {x, y}]
     = {Sqrt[2] InverseErfc[2 - 2 x], Sqrt[2] InverseErfc[2 - 2 y]}

    >> InverseCDF[NormalDistribution[0, 1]]
     = Sqrt[2] InverseErfc[2 - 2 #1]&
    '''

    rules = {
        'InverseCDF[dist_, x_]': 'InverseCDF[dist][x]',
        'InverseCDF[dist_, xs_List]': 'InverseCDF[dist] /@ xs',
    }

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
    <dl>
    <dt>'NormalDistribution[$mu$, $sigma$]'
      <dd>represents the normal distribution with mean $mu$ and standard deviation $sigma$.
    <dt>'NormalDistribution[]'
      <dd>represents the normal distribution with mean zero and standard deviation one.
    </dl>

    >> CDF[NormalDistribution[mu, sigma]]
     = 1 / 2 + Erf[Sqrt[2] (-mu + #1) / (2 sigma)] / 2&

    >> PDF[NormalDistribution[0, 1]]
     = Sqrt[2] Exp[-#1 ^ 2 / 2] / (2 Sqrt[Pi])&

    >> Mean[NormalDistribution[mu, sigma]]
     = mu

    >> Variance[NormalDistribution[mu, sigma]]
     = sigma ^ 2

    >> Plot[PDF[NormalDistribution[], x], {x, -6, 6}, PlotRange->All]
     = -Graphics-

    #> CDF[NormalDistribution[0, 1]]
     = 1 - Erfc[Sqrt[2] #1 / 2] / 2&
    '''

    params = (RealParam, PositiveRealParam)
    sympy_name = 'Normal'

    rules = {
        'NormalDistribution[]': 'NormalDistribution[0, 1]',
    }
