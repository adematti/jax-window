from functools import partial
from dataclasses import dataclass, field

import numpy as np
import jax
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


import os
from jaxpower import BinnedStatistic, WindowMatrix, utils


def _ravel_index(observable, iproj=0, ix=Ellipsis):
    sizes = [len(xx) for xx in observable._x]
    return sum(sizes[:iproj]) + (np.arange(sizes[iproj]) if ix is Ellipsis else np.array(ix))


def batch_jacrev(func, theory, seed=42, nmocks=None, indices=(Ellipsis, Ellipsis), batchs=1, callback=None, func_kwargs=None, jit=True):
    from tqdm import tqdm

    if np.ndim(batchs) == 0: batchs = (1, batchs)
    kw = dict(func_kwargs or {})
    run_mocks = nmocks is not None
    if run_mocks:
        if isinstance(seed, int):
            seed = random.key(seed)
        kw.update(seed=seed)
    else:
        nmocks = 1

    observable = func(theory, **kw)
    has_aux = isinstance(observable, tuple)
    if has_aux: observable = observable[0]

    state = None
    if callback is None:
        state = dict(primals=[], tangents=np.zeros((observable.size, theory.size), dtype=float), aux=[])

        def callback(iproj, ix, primals, tangents, aux, state=state):
            state['primals'].append(primals)
            state['tangents'][_ravel_index(observable, iproj=iproj, ix=ix)] += tangents
            state['aux'].append(aux)

    iprojs = indices[0]
    if iprojs is Ellipsis: iprojs = list(range(len(theory.projs)))
    nsplits = (len(iprojs) + batchs[0] - 1) // batchs[0]
    splits_iprojs = [iprojs[i * len(iprojs) // nsplits:(i + 1) * len(iprojs) // nsplits] for i in range(nsplits)]
    ixs = [np.arange(len(xx))[indices[1]] if iproj in iprojs else [] for iproj, xx in enumerate(observable._x)]

    def func_aux(value, **kw):
        th = theory.clone(value=value)
        observable = func(th, **kw)
        aux = None
        if has_aux:
            observable, aux = observable
        observable = observable.view().real
        return observable, aux

    def vjp(tangent, **kw):
        tmp = jax.vjp(partial(func_aux, **kw), theory.view(), has_aux=True)
        vjp_fun = tmp[1]
        tangent = jax.vmap(vjp_fun, in_axes=0)(tangent)
        #print(tangent)
        return tmp[:1] + tangent + tmp[2:]

    if jit:
        vjp = jax.jit(vjp)

    zeros = jnp.zeros_like(observable.view().real)

    def get_basis(iprojs, ix):
        return jnp.array([zeros.at[_ravel_index(observable, iproj=iiproj, ix=iix)].set(1.) for iiproj in iprojs for iix in ix])

    total = sum(len(ix) for ix in ixs) * nmocks
    t = tqdm(total=total)
    for iprojs in splits_iprojs:
        ix = ixs[iprojs[0]]
        nsplits = (len(ix) + batchs[1] - 1) // batchs[1]
        splits_ix = [ix[i * len(ix) // nsplits:(i + 1) * len(ix) // nsplits] for i in range(nsplits)]
        for ix in splits_ix:
            for imock in range(nmocks):
                if run_mocks:
                    seed2, seed = random.split(seed)
                    kw.update(seed=seed2)
                res = vjp(get_basis(iprojs, ix), **kw)
                jax.block_until_ready(res)
                for iproj in iprojs:
                    tmp = list(res)
                    tmp[1] = tmp[1].reshape(len(iprojs), len(ix), *tmp[1].shape[1:])[iproj]
                    callback(iproj, ix, *tmp)
                t.update(n=len(iprojs) * len(ix))

    return state


def batch_jacfwd(func, theory, seed=42, nmocks=None, indices=(Ellipsis, Ellipsis), batchs=1, callback=None, func_kwargs=None, linear=False, jit=True):
    from tqdm import tqdm
    if np.ndim(batchs) == 0: batchs = (1, batchs)
    kw = dict(func_kwargs or {})
    run_mocks = nmocks is not None
    if run_mocks:
        if isinstance(seed, int):
            seed = random.key(seed)
        kw.udpate(seed=seed)
    else:
        nmocks = 1
    observable = func(theory, **kw)
    has_aux = isinstance(observable, tuple)
    if has_aux: observable = observable[0]

    state = None
    if callback is None:
        state = dict(primals=[], tangents=np.zeros((observable.size, theory.size), dtype=float), aux=[])

        def callback(iproj, ix, primals, tangents, aux, state=state):
            state['primals'].append(primals)
            state['tangents'][..., _ravel_index(theory, iproj=iproj, ix=ix)] += tangents.T
            state['aux'].append(aux)

    iprojs = indices[0]
    if iprojs is Ellipsis: iprojs = list(range(len(theory.projs)))
    nsplits = (len(iprojs) + batchs[0] - 1) // batchs[0]
    splits_iprojs = [iprojs[i * len(iprojs) // nsplits:(i + 1) * len(iprojs) // nsplits] for i in range(nsplits)]
    ixs = [np.arange(len(xx))[indices[1]] if iproj in iprojs else [] for iproj, xx in enumerate(theory._x)]

    def func_aux(value, **kw):
        th = theory.clone(value=value)
        observable = func(th, **kw)
        aux = None
        if has_aux:
            observable, aux = observable
        observable = observable.view().real
        return observable, aux

    def jvp(tangent, **kw):
        func_aux2 = lambda tangent: (observable.view().real,) + func_aux(tangent * (1 + 1e-9), **kw)
        if linear:
            if tangent.shape[0] == 1:
                toret = func_aux2(tangent[0])
                toret = tuple(tmp[None, ...] for tmp in toret[:2]) + toret[2:]
            else:
                toret = jax.vmap(func_aux2, in_axes=0)(tangent)
        else:
            toret = jax.vmap(lambda tangent: jax.jvp(partial(func_aux, **kw), (theory.view(),), (tangent,), has_aux=True), in_axes=0)(tangent)
        return (toret[0][0],) + toret[1:]

    if jit:
        jvp = jax.jit(jvp)

    zeros = jnp.zeros_like(theory.view())

    def get_basis(iprojs, ix):
        return jnp.array([zeros.at[_ravel_index(theory, iproj=iproj, ix=iix)].set(1.) for iproj in iprojs for iix in ix])

    total = sum(len(ix) for ix in ixs) * nmocks
    t = tqdm(total=total)
    for iprojs in splits_iprojs:
        ix = ixs[iprojs[0]]
        nsplits = (len(ix) + batchs[1] - 1) // batchs[1]
        splits_ix = [ix[i * len(ix) // nsplits:(i + 1) * len(ix) // nsplits] for i in range(nsplits)]
        for ix in splits_ix:
            for imock in range(nmocks):
                if run_mocks:
                    seed2, seed = random.split(seed)
                    kw.update(seed=seed2)
                res = jvp(get_basis(iprojs, ix), **kw)
                jax.block_until_ready(res)
                for iproj in iprojs:
                    tmp = list(res)
                    tmp[1] = tmp[1].reshape(len(iprojs), len(ix), *tmp[1].shape[1:])[iproj]
                    callback(iproj, ix, *tmp)
                t.update(n=len(iprojs) * len(ix))

    return state


@dataclass
class WindowMatrixEstimator(object):

    """Data class to estimate the window matrix."""

    theory: BinnedStatistic = None
    observable: BinnedStatistic = field(default=None, init=False)
    _meta_fields = ['theory', 'observable']
    _data_fields = ['wmat_wsum', 'wmat_nsum', 'wmat_theory_samples', 'observable_samples', 'wmat_cv', 'observable_cv']

    def reset(self):
        self.wmat_wsum = np.zeros((self.observable.size, self.theory.size), dtype=float)
        self.wmat_nsum = np.zeros_like(self.wmat_wsum, dtype=int)
        self.wmat_theory_samples = [[] for idx in range(self.observable.size)]
        self.observable_samples = []
        self.wmat_cv = self.wmat_wsum.copy()
        self.observable_cv = np.zeros(self.observable.size, dtype=float)

    def clone(self, **kwargs):
        import copy
        """Create a new instance, updating some attributes."""
        state = {name: getattr(self, name).clone() for name in self._meta_fields if getattr(self, name, None) is not None}
        state.update(copy.deepcopy({name: getattr(self, name) for name in self._data_fields  if hasattr(self, name)}))
        state.update(kwargs)
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(state)
        return new

    def __getstate__(self):
        state = {name: getattr(self, name).__getstate__() for name in self._meta_fields if getattr(self, name, None) is not None}
        state.update({name: getattr(self, name) for name in self._data_fields})
        return state

    def __setstate__(self, state):
        state = dict(state)
        for name in self._meta_fields:
            if state.get(name, None) is not None:
                state[name] = BinnedStatistic.from_state(state[name])
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        """Save object."""
        state = self.__getstate__()
        #self.log_info('Saving {}.'.format(filename))
        utils.mkdir(os.path.dirname(filename))
        np.save(filename, state, allow_pickle=True)

    @classmethod
    def load(cls, filename):
        """Load object."""
        filename = str(filename)
        state = np.load(filename, allow_pickle=True)
        #cls.log_info('Loading {}.'.format(filename))
        state = state[()]
        return cls.from_state(state)

    def cv(self, func, mode='fwd', batchs=1, func_kwargs=None, **kwargs):
        observable = func(self.theory, **(func_kwargs or {}))
        if self.observable is None:
            self.observable = observable
            self.reset()
        batch_jac = partial(batch_jacfwd, linear=True) if mode == 'fwd' else batch_jacrev
        state = batch_jac(func, self.theory, batchs=batchs, func_kwargs=func_kwargs, **kwargs)
        self.wmat_cv = state['tangents']
        self.observable_cv = state['primals'][0]

    def sample(self, func, mode='rev', seed=random.key(42), nmocks=25, indices=(Ellipsis, Ellipsis), batchs=1, func_kwargs=None, **kwargs):
        if self.observable is None:
            self.observable = func(self.theory, seed=seed, **(func_kwargs or {}))
            self.reset()

        if mode == 'fwd':
            batch_jac = batch_jacfwd

            def callback(iproj, ix, primals, tangents, aux=None):
                idx = _ravel_index(self.theory, iproj=iproj, ix=ix)
                for idx, tangent in zip(idx, tangents):
                    self.wmat_wsum[..., idx] += tangent
                    self.wmat_nsum[..., idx] += 1
                self.observable_samples.append(primals)

        else:
            batch_jac = batch_jacrev

            def callback(iproj, ix, primals, tangents, aux=None):
                idx = _ravel_index(self.observable, iproj=iproj, ix=ix)
                for idx, tangent in zip(idx, tangents):
                    self.wmat_wsum[idx] += tangent
                    self.wmat_nsum[idx] += 1
                    self.wmat_theory_samples[idx].append(tangent.dot(self.theory.view()))
                self.observable_samples.append(primals)

        batch_jac(func, self.theory, seed=seed, nmocks=nmocks, indices=indices, batchs=batchs, callback=callback, func_kwargs=func_kwargs, **kwargs)

    def mean(self, interp=False, **kwargs):
        if interp:
            wmat = self.wmat_wsum / np.where(self.wmat_nsum == 0, 1, self.wmat_nsum)
            from scipy import interpolate
            # x is observable.x, y is theory.x - x, z is wmat
            for ipo, po in enumerate(self.observable.projs):
                for ipt, pt in enumerate(self.theory.projs):
                    x, y, z = [], [], []
                    for ixo, xo in enumerate(self.observable.x(projs=po)):
                        for ixt, xt in enumerate(self.theory.x(projs=pt)):
                            idx = _ravel_index(self.observable, ix=ixo, iproj=ipo), _ravel_index(self.theory, ix=ixt, iproj=ipt)
                            if self.wmat_nsum[idx] > 0:
                                x.append(xo)
                                y.append(xt - xo)
                                z.append(wmat[idx])
                    interp = interpolate.LinearNDInterpolator(np.column_stack([x, y]), z, fill_value=0., rescale=False)
                    for ixo, xo in enumerate(self.observable.x(projs=po)):
                        idx = _ravel_index(self.observable, ix=ixo, iproj=ipo), _ravel_index(self.theory, ix=Ellipsis, iproj=ipt)
                        wmat[idx] = interp(xo, self.theory.x(projs=pt) - xo)
        else:
            wmat = self.wmat_wsum / np.where(self.wmat_nsum == 0, 1, self.wmat_nsum)
        wmat += self.wmat_cv
        value = np.mean(self.observable_samples, axis=0) if self.observable_samples else 0
        value += self.observable_cv
        observable = self.observable.clone(value=value)
        return WindowMatrix(observable=observable, theory=self.theory, value=wmat)

    def std(self, interp=False, std_on_mean=False):
        std = []
        for wt in self.wmat_theory_samples:
            if len(wt):
                std.append(np.std(wt, ddof=1) / (np.sqrt(len(wt)) if std_on_mean else 1.))
            else:
                std.append(np.nan)
        std = np.array(std)
        toret = self.observable.clone(value=std)
        if interp:
            std = []
            for iproj, proj in enumerate(toret.projs):
                x, v = toret._x[iproj], toret._value[iproj]
                if not np.isnan(v).all():
                    mask = ~np.isnan(v)
                    v = np.interp(x, x[mask], v[mask], left=0., right=0.)
                std.append(v)
            toret = toret.clone(value=std)
        return toret
