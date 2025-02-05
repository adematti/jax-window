import numpy as np
from jax import random
from jax import numpy as jnp


def generate_acceptable_poles(power0, alpha1=None, alpha2=None, seed=42):
    if isinstance(seed, int):
        seed = random.key(seed)
    seeds = random.split(seed)
    if alpha1 is None:
        alpha1 = random.uniform(seeds[0], minval=0.05, maxval=0.1)
    power4 = alpha1 * power0  # hexadecapole ~ 1 / 10 of monopole
    a2sq = 35. / 18. * power4
    a0sq = power0 - 1. / 5. * a2sq
    minval = np.max(- 1. / 7. * a2sq / (a0sq * a2sq)**0.5)
    if alpha2 is None:
        alpha2 = random.uniform(seeds[1], minval=minval, maxval=1.)
    alpha2 = np.clip(alpha2, minval, 1.)
    power2 = 2 * alpha2 * (a0sq * a2sq)**0.5 + 2. / 7. * a2sq
    return [power0, power2, power4]


import math

from fractions import Fraction


def wigner3j_square(ellout, ellin, prefactor=True):
    r"""
    Return the coefficients corresponding to the product of two Legendre polynomials, corresponding to :math:`C_{\ell \ell^{\prime} L}`
    of e.g. eq. 2.2 of https://arxiv.org/pdf/2106.06324.pdf, with :math:`\ell` corresponding to ``ellout`` and :math:`\ell^{\prime}` to ``ellin``.

    Parameters
    ----------
    ellout : int
        Output order.

    ellin : int
        Input order.

    prefactor : bool, default=True
        Whether to include prefactor :math:`(2 \ell + 1)/(2 L + 1)` for window convolution.

    Returns
    -------
    ells : list
        List of mulipole orders :math:`L`.

    coeffs : list
        List of corresponding window coefficients.
    """
    qvals, coeffs = [], []

    def G(p):
        """
        Return the function G(p), as defined in Wilson et al 2015.
        See also: WA Al-Salam 1953
        Taken from https://github.com/nickhand/pyRSD.

        Parameters
        ----------
        p : int
            Multipole order.

        Returns
        -------
        numer, denom: int
            The numerator and denominator.
        """
        toret = 1
        for p in range(1, p + 1): toret *= (2 * p - 1)
        return toret, math.factorial(p)

    for p in range(min(ellin, ellout) + 1):

        numer, denom = [], []

        # numerator of product of G(x)
        for r in [G(ellout - p), G(p), G(ellin - p)]:
            numer.append(r[0])
            denom.append(r[1])

        # divide by this
        a, b = G(ellin + ellout - p)
        numer.append(b)
        denom.append(a)

        numer.append(2 * (ellin + ellout) - 4 * p + 1)
        denom.append(2 * (ellin + ellout) - 2 * p + 1)

        q = ellin + ellout - 2 * p
        if prefactor:
            numer.append(2 * ellout + 1)
            denom.append(2 * q + 1)

        coeffs.append(Fraction(np.prod(numer, dtype='i8'), np.prod(denom, dtype='i8')))
        qvals.append(q)

    return qvals[::-1], coeffs[::-1]
