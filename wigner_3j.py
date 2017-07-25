"""
Sources:
    
    http://docs.sympy.org/latest/modules/physics/wigner.html
    http://docs.sympy.org/dev/_modules/sympy/physics/wigner.html
    
"""
from __future__ import print_function, division

from sympy import (Integer, pi, sqrt, sympify, Dummy, S, Sum, Ynm,
        Function)
from sympy.core.compatibility import range

_Factlist = [1]


def _calc_factlist(nn):
    r"""
    Function calculates a list of precomputed factorials in order to
    massively accelerate future calculations of the various
    coefficients.

    INPUT:

    -  ``nn`` -  integer, highest factorial to be computed

    OUTPUT:

    list of integers -- the list of precomputed factorials

    EXAMPLES:

    Calculate list of factorials::

        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """
    if nn >= len(_Factlist):
        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)
    return _Factlist[:int(nn) + 1]

def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

    INPUT:

    -  ``j_1``, ``j_2``, ``j_3``, ``m_1``, ``m_2``, ``m_3`` - integer or half integer

    OUTPUT:

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    It is an error to have arguments that are not integer or half
    integer values::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    NOTES:

    The Wigner 3j symbol obeys the following symmetry rules:

    - invariant under any permutation of the columns (with the
      exception of a sign change where `J:=j_1+j_2+j_3`):

      .. math::

         \begin{aligned}
         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
          &=\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\
          &=\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\
          &=(-1)^J \operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
         \end{aligned}

    - invariant under space inflection, i.e.

      .. math::

         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
         =(-1)^J \operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 additional symmetries based on
      the work by [Regge58]_

    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation

    - zero for `m_1 + m_2 + m_3 \neq 0`

    - zero for violating any one of the conditions
      `j_1 \ge |m_1|`,  `j_2 \ge |m_2|`,  `j_3 \ge |m_3|`

    ALGORITHM:

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    REFERENCES:

    .. [Regge58] 'Symmetry Properties of Clebsch-Gordan Coefficients',
      T. Regge, Nuovo Cimento, Volume 10, pp. 544 (1958)

    .. [Edmonds74] 'Angular Momentum in Quantum Mechanics',
      A. R. Edmonds, Princeton University Press (1974)

    AUTHORS:

    - Jens Rasch (2009-03-24): initial version
    """
    if int(j_1 * 2) != j_1 * 2 or int(j_2 * 2) != j_2 * 2 or \
            int(j_3 * 2) != j_3 * 2:
        raise ValueError("j values must be integer or half integer")
    if int(m_1 * 2) != m_1 * 2 or int(m_2 * 2) != m_2 * 2 or \
            int(m_3 * 2) != m_3 * 2:
        raise ValueError("m values must be integer or half integer")
    if m_1 + m_2 + m_3 != 0:
        return 0
    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    m_3 = -m_3
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return 0
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return 0
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return 0
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return 0

    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),
                  j_3 + abs(m_3))
    _calc_factlist(int(maxfact))

    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] *
                     _Factlist[int(j_1 - j_2 + j_3)] *
                     _Factlist[int(-j_1 + j_2 + j_3)] *
                     _Factlist[int(j_1 - m_1)] *
                     _Factlist[int(j_1 + m_1)] *
                     _Factlist[int(j_2 - m_2)] *
                     _Factlist[int(j_2 + m_2)] *
                     _Factlist[int(j_3 - m_3)] *
                     _Factlist[int(j_3 + m_3)]) / \
        _Factlist[int(j_1 + j_2 + j_3 + 1)]

    ressqrt = sqrt(argsqrt)
    if ressqrt.is_complex:
        ressqrt = ressqrt.as_real_imag()[0]

    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * \
            _Factlist[int(ii + j_3 - j_1 - m_2)] * \
            _Factlist[int(j_2 + m_2 - ii)] * \
            _Factlist[int(j_1 - ii - m_1)] * \
            _Factlist[int(ii + j_3 - j_2 + m_1)] * \
            _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den

    res = ressqrt * sumres * prefid
    return res

def gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    r"""
    Calculate the Gaunt coefficient.

    The Gaunt coefficient is defined as the integral over three
    spherical harmonics:

    .. math::

        \begin{aligned}
        \operatorname{Gaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Y_{l_1,m_1}(\Omega)
         Y_{l_2,m_2}(\Omega) Y_{l_3,m_3}(\Omega) \,d\Omega \\
        &=\sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
         \operatorname{Wigner3j}(l_1,l_2,l_3,0,0,0)
         \operatorname{Wigner3j}(l_1,l_2,l_3,m_1,m_2,m_3)
        \end{aligned}

    INPUT:

    -  ``l_1``, ``l_2``, ``l_3``, ``m_1``, ``m_2``, ``m_3`` - integer

    -  ``prec`` - precision, default: ``None``. Providing a precision can
       drastically speed up the calculation.

    OUTPUT:

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import gaunt
    >>> gaunt(1,0,1,1,0,-1)
    -1/(2*sqrt(pi))
    >>> gaunt(1000,1000,1200,9,3,-12).n(64)
    0.00689500421922113448...

    It is an error to use non-integer values for `l` and `m`::

        sage: gaunt(1.2,0,1.2,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer
        sage: gaunt(1,0,1,1.1,0,-1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer

    NOTES:

    The Gaunt coefficient obeys the following symmetry rules:

    - invariant under any permutation of the columns

      .. math::
        \begin{aligned}
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          &=Y(l_3,l_1,l_2,m_3,m_1,m_2) \\
          &=Y(l_2,l_3,l_1,m_2,m_3,m_1) \\
          &=Y(l_3,l_2,l_1,m_3,m_2,m_1) \\
          &=Y(l_1,l_3,l_2,m_1,m_3,m_2) \\
          &=Y(l_2,l_1,l_3,m_2,m_1,m_3)
        \end{aligned}

    - invariant under space inflection, i.e.

      .. math::
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          =Y(l_1,l_2,l_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 Regge symmetries as inherited
      for the `3j` symbols [Regge58]_

    - zero for `l_1`, `l_2`, `l_3` not fulfilling triangle relation

    - zero for violating any one of the conditions: `l_1 \ge |m_1|`,
      `l_2 \ge |m_2|`, `l_3 \ge |m_3|`

    - non-zero only for an even sum of the `l_i`, i.e.
      `L = l_1 + l_2 + l_3 = 2n` for `n` in `\mathbb{N}`

    ALGORITHM:

    This function uses the algorithm of [Liberatodebrito82]_ to
    calculate the value of the Gaunt coefficient exactly. Note that
    the formula contains alternating sums over large factorials and is
    therefore unsuitable for finite precision arithmetic and only
    useful for a computer algebra system [Rasch03]_.

    REFERENCES:

    .. [Liberatodebrito82] 'FORTRAN program for the integral of three
      spherical harmonics', A. Liberato de Brito,
      Comput. Phys. Commun., Volume 25, pp. 81-85 (1982)

    AUTHORS:

    - Jens Rasch (2009-03-24): initial version for Sage
    """
    if int(l_1) != l_1 or int(l_2) != l_2 or int(l_3) != l_3:
        raise ValueError("l values must be integer")
    if int(m_1) != m_1 or int(m_2) != m_2 or int(m_3) != m_3:
        raise ValueError("m values must be integer")

    sumL = l_1 + l_2 + l_3
    bigL = sumL // 2
    a1 = l_1 + l_2 - l_3
    if a1 < 0:
        return 0
    a2 = l_1 - l_2 + l_3
    if a2 < 0:
        return 0
    a3 = -l_1 + l_2 + l_3
    if a3 < 0:
        return 0
    if sumL % 2:
        return 0
    if (m_1 + m_2 + m_3) != 0:
        return 0
    if (abs(m_1) > l_1) or (abs(m_2) > l_2) or (abs(m_3) > l_3):
        return 0

    imin = max(-l_3 + l_1 + m_2, -l_3 + l_2 - m_1, 0)
    imax = min(l_2 + m_2, l_1 - m_1, l_1 + l_2 - l_3)

    maxfact = max(l_1 + l_2 + l_3 + 1, imax + 1)
    _calc_factlist(maxfact)

    argsqrt = (2 * l_1 + 1) * (2 * l_2 + 1) * (2 * l_3 + 1) * \
        _Factlist[l_1 - m_1] * _Factlist[l_1 + m_1] * _Factlist[l_2 - m_2] * \
        _Factlist[l_2 + m_2] * _Factlist[l_3 - m_3] * _Factlist[l_3 + m_3] / \
        (4*pi)
    ressqrt = sqrt(argsqrt)

    prefac = Integer(_Factlist[bigL] * _Factlist[l_2 - l_1 + l_3] *
                     _Factlist[l_1 - l_2 + l_3] * _Factlist[l_1 + l_2 - l_3])/ \
        _Factlist[2 * bigL + 1]/ \
        (_Factlist[bigL - l_1] *
         _Factlist[bigL - l_2] * _Factlist[bigL - l_3])

    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * _Factlist[ii + l_3 - l_1 - m_2] * \
            _Factlist[l_2 + m_2 - ii] * _Factlist[l_1 - ii - m_1] * \
            _Factlist[ii + l_3 - l_2 + m_1] * _Factlist[l_1 + l_2 - l_3 - ii]
        sumres = sumres + Integer((-1) ** ii) / den

    res = ressqrt * prefac * sumres * (-1) ** (bigL + l_3 + m_1 - m_2)
    if prec is not None:
        res = res.n(prec)
    return res



class Wigner3j(Function):

    def doit(self, **hints):
        if all(obj.is_number for obj in self.args):
            return wigner_3j(*self.args)
        else:
            return self

"""
THESE ARE JUST USED BY IMPORTING FROM SYMPY
"""

from sympy.physics.wigner import wigner_3j
wigner_3j(2, 6, 4, 0, 0, 0)

from sympy.physics,wigner import gaunt