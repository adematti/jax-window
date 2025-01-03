import itertools

import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax import random, jit

from . import utils
from .utils import _make_array


class BaseDensity(object):
    
    @property
    def meshsize(self):
        return np.array(self.density.shape)

    @property
    def cellsize(self):
        return self.boxsize / self.meshsize

    @property
    def cellsize(self):
        return self.boxsize / self.meshsize

    def tree_flatten(self):
        return ({name: getattr(self, name) for name in ['density', 'boxsize']},)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        di, = children
        new.__dict__.update(di)
        return new


@register_pytree_node_class
class GaussianField(BaseDensity):

    def __init__(self, power, seed=42, boxsize=1000., meshsize=128, unitary_amplitude=False):
        if isinstance(seed, int):
            seed = random.key(seed)

        ndim = 3
        for value in [boxsize, meshsize]:
            try: ndim = len(value)
            except: pass
        self.boxsize = _make_array(boxsize, ndim)
        meshsize = _make_array(meshsize, ndim)
        cellsize = self.boxsize / meshsize
        shape = tuple(meshsize)
        kmesh = utils.fftfreq(shape, cellsize)
        kmesh = sum(kk**2 for kk in kmesh)**0.5
        pkmesh = power(kmesh.ravel()).reshape(kmesh.shape) / cellsize.prod()

        field = random.normal(seed, shape)
        field = jnp.fft.rfftn(field)
        if unitary_amplitude:
            field *= meshsize.prod()**0.5 / jnp.abs(field)
        self.density = 1. + jnp.fft.irfftn(field * pkmesh**0.5)


@jit
def cic_paint(field, positions, values=1.):
    boxsize = jnp.array(field.shape)

    def wrap(idx):
        return idx % jnp.array(field.shape)

    fidx = positions % boxsize
    idx = jnp.floor(fidx).astype('i4')
    dx = fidx - idx

    for ishift in itertools.product(*([0, 1],) * 3):
        ishift = np.array(ishift)
        field = field.at[tuple(wrap(idx + ishift).T)].add(values * jnp.prod(dx * (ishift == 1) + (1. - dx) * (ishift == 0), axis=-1))
    return field
    

@register_pytree_node_class
class SurveySelection(BaseDensity):

    def __init__(self, positions, weights=1., cellsize=10., boxsize=None, meshsize=None, boxcenter=None, boxpad=1.2):
        ndim = positions.shape[-1]
        extent = np.min(positions, axis=0), np.max(positions, axis=0)
        cellsize, boxsize, meshsize, boxcenter = map(lambda value: _make_array(value, ndim) if value is not None else None, [cellsize, boxsize, meshsize, boxcenter])
        if boxsize is None:
            if meshsize is not None and cellsize is not None:
                boxsize = cellsize * meshsize
            boxsize = (extent[-1] - extent[0]) * boxpad
        if meshsize is None:
            if cellsize is None:
                raise ValueError('provide meshsize or cellsize')
            meshsize = np.rint(boxsize / cellsize).astype(int)
        if boxcenter is None:
            boxcenter = np.mean(extent, axis=0)
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.density = jnp.zeros(tuple(meshsize), dtype=positions.dtype)
        self.density = cic_paint(self.density, (positions - self.boxcenter + self.boxsize / 2.) / self.cellsize, values=weights)

    def tree_flatten(self):
        return ({name: getattr(self, name) for name in ['density', 'boxsize', 'boxcenter']},)
