import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from . import utils
from .utils import _make_array


@register_pytree_node_class
class BoxFFTPower(object):

    def __init__(self, field, cellsize, edges, field2=None, compensate=False):
        dtype, shape = field.dtype, field.shape
        meshsize = np.array(shape)
        cellsize = _make_array(cellsize, len(shape))
        kfun = np.min(2 * np.pi / (cellsize * meshsize))
        knyq = np.min(np.pi / cellsize)

        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            kmin = edges.get('min', 0.)
            kmax = edges.get('max', knyq)
            kstep = edges.get('step', kfun)
            edges = np.arange(kmin, kmax, kstep)
        else:
            edges = np.asarray(edges)

        field = jnp.fft.rfftn(field, norm=None)
        if field2 is None:
            power = field.real**2 + field.imag**2
        else:
            field2 = jnp.fft.rfftn(field2, norm=None)
            power = field * field2.conj()

        kvec = utils.fftfreq(shape, cellsize)
        k = jnp.sqrt(sum(kk.astype(dtype)**2 for kk in kvec))

        if compensate:
            compensate = {'ngc': 2, 'cic': 4, 'tsc': 6, 'pcs': 8}.get(compensate, compensate)
            for kk, cc in zip(kvec, cellsize):
                power *= jnp.sinc(0.5 / np.pi * kk * cc)**-compensate

        nmodes = jnp.full_like(power, 2, dtype='i4')
        nmodes = nmodes.at[..., 0].set(1)
        if shape[-1] % 2 == 0:
            nmodes = nmodes.at[..., -1].set(1)

        k = k.ravel()
        power = power.ravel()
        nmodes = nmodes.ravel()
        ibin = jnp.digitize(k, edges, right=False)
        k = (k * nmodes).astype(dtype)
        power = (power * nmodes).astype(dtype)
        k = jnp.bincount(ibin, weights=k, length=edges.size + 1)[1:-1]
        power = jnp.bincount(ibin, weights=power, length=edges.size + 1)[1:-1]
        nmodes = jnp.bincount(ibin, weights=nmodes, length=edges.size + 1)[1:-1]

        k /= nmodes
        power *= jnp.prod(cellsize / meshsize) / nmodes
        self.k, self.power, self.nmodes, self.edges, self.cellsize = k, power, nmodes, edges, cellsize

    def tree_flatten(self):
        return ({name: getattr(self, name) for name in ['k', 'power', 'nmodes', 'edges', 'cellsize']},), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        di, = children
        new.__dict__.update(di)
        return new

    def save(self, fn):
        fn = str(fn)
        mkdir(os.path.dirname(fn))
        np.save(fn, self.tree_flatten(), allow_pickle=True)

    @classmethod
    def load(cls, fn):
        fn = str(fn)
        aux_data, children = np.load(fn, allow_pickle=True)
        return cls.tree_unflatten(aux_data, children)


@register_pytree_node_class
class SurveyFFTPower(BoxFFTPower):

    def __init__(self, data, randoms, cellsize, edges, data2=None, randoms2=None, compensate=False, norm=None, shotnoise_nonorm=0.):

        alpha = jnp.mean(data) / jnp.mean(randoms)
        field = data - alpha * randoms
        if data2 is None:
            field2 = None
            if norm is None: norm = alpha * jnp.mean(data * randoms)
            field /= norm**0.5
        else:
            if randoms2 is None:
                randoms2 = randoms
            alpha2 = jnp.mean(data2) / jnp.mean(randoms2)
            field2 = data2 - alpha2 * randoms2
            if norm is None: norm = (alpha2 * jnp.mean(data * randoms2) + alpha * jnp.mean(data2 * randoms)) / 2.
            field /= norm**0.5
            field2 /= norm**0.5
        super().__init__(field, cellsize, edges, field2=field2, compensate=compensate)
        self.power -= shotnoise_nonorm / norm * jnp.prod(self.cellsize) / np.prod(field.shape)
        