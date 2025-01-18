import numpy as np
from jax import random
from jax import numpy as jnp

from collections.abc import Callable
from typing import Union

from jaxpower.mesh import RealMeshField, staticarray
from jaxpower.mock import _get_ndim
from jaxpower.power import _get_los_vector, legendre, get_real_Ylm



def generate_acceptable_poles(power0, seed=42):
    if isinstance(seed, int):
        seed = random.key(seed)
    seeds = random.split(seed)
    power4 = random.uniform(seeds[0], minval=0.05, maxval=0.1) * power0  # hexadecapole ~ 1 / 10 of monopole
    a2sq = 35. / 18. * power4
    a0sq = power0 - 1. / 5. * a2sq
    minval = - 1. / 7. * a2sq / (a0sq * a2sq)**0.5
    power2 = 2 * random.uniform(seeds[1], minval=minval, maxval=1.) * (a0sq * a2sq)**0.5 + 2. / 7. * a2sq
    return [power0, power2, power4]


def generate_anisotropic_gaussian_mesh(k, poles: dict[Callable], los: str='x',
                                       boxsize: Union[float, np.ndarray]=1000., meshsize: Union[int, np.ndarray]=128,
                                       unitary_amplitude: bool=False, boxcenter=0., seed: int=42):

    """Generate :class:`RealMeshField` with input power."""

    ells = (0, 2, 4)

    if isinstance(seed, int):
        seed = random.key(seed)

    ndim = _get_ndim(boxsize, meshsize, boxcenter)
    shape = staticarray.fill(meshsize, ndim)

    def _safe_divide(num, denom):
        with np.errstate(divide='ignore', invalid='ignore'):
            return jnp.where(denom == 0., 0., num / denom)


    if los == 'local':
        seeds = random.split(seed)
        meshs = [RealMeshField(random.normal(seed, shape), boxsize=boxsize, boxcenter=boxcenter).r2c() for seed in seeds]
        meshsize, cellsize = meshs[0].meshsize, meshs[0].cellsize
        if unitary_amplitude:
            for imesh, mesh in enumerate(meshs):
                meshs[imesh] *= meshsize.prod()**0.5 / jnp.abs(mesh.value)

        a11 = 35. / 18. * poles[2] / cellsize.prod()
        a00 = poles[0] / cellsize.prod() - 1. / 5. * a11
        a10 = 1. / 2. * poles[1] / cellsize.prod() - 1. / 7. * a11

        # Cholesky decomposition
        l00 = a00**0.5
        l10 = a10 / l00
        l11 = (a11 - l10**2)**0.5
        #meshs[1] = l10 * meshs[0] + l11 * meshs[1]
        #meshs[0] = l00 * meshs[0]

        kvec = meshs[0].coords(sparse=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))

        def _interp(pk):
            return jnp.interp(knorm.ravel(), k, pk, left=0., right=0.).reshape(meshs[0].shape)

        # The mesh for ell = 0
        mesh = meshs[0] * _interp(l00)
        mesh = mesh.c2r()

        # The mesh for ell = 2
        mesh2 = meshs[0] * _interp(l10) + meshs[1] * _interp(l11)
        # Now let's take care of L2
        del meshs

        # The Fourier-space grid
        khat = [_safe_divide(kk, knorm) for kk in kvec]
        del knorm

        # The real-space grid
        xhat = mesh.coords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in xhat))
        xhat = [_safe_divide(xx, xnorm) for xx in xhat]
        del xnorm

        ell = 2
        Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
        mesh += 4. * np.pi / (2 * ell + 1) * sum((mesh2 * Ylm(*khat)).c2r() * Ylm(*xhat) for Ylm in Ylms)  # total mesh, mesh0 + mesh2 * L2(mu)

        return mesh

    else:
        vlos = _get_los_vector(los, ndim=ndim)
        mesh = RealMeshField(random.normal(seed, shape), boxsize=boxsize, boxcenter=boxcenter).r2c()
        meshsize, cellsize = mesh.meshsize, mesh.cellsize

        def kernel(value, kvec):
            knorm = jnp.sqrt(sum(kk**2 for kk in kvec)).ravel()
            mu = _safe_divide(sum(kk * ll for kk, ll in zip(kvec, vlos)).ravel(), knorm)
            ker = 0.
            for ill, ell in enumerate(ells):
                ker += jnp.interp(knorm, k, poles[ill] / cellsize.prod(), left=0., right=0.) * legendre(ell)(mu)
            ker = jnp.sqrt(ker).reshape(value.shape)
            if unitary_amplitude:
                ker *= meshsize.prod()**0.5 / jnp.abs(value)
            return value * ker

        mesh = mesh.apply(kernel, kind='wavenumber')
        return mesh.c2r()


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
