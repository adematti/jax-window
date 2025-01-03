from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp
from jax import random

from matplotlib import pyplot as plt

from jpower import GaussianField, SurveySelection, BoxFFTPower, SurveyFFTPower, utils


dirname = Path('_tests')


def mock_box(power, unitary_amplitude=True, **kwargs):
    field = GaussianField(power, unitary_amplitude=unitary_amplitude, **kwargs)
    edges = {'step': 0.01}
    return BoxFFTPower(field.density / jnp.mean(field.density) - 1., cellsize=field.cellsize, edges=edges, compensate=None)


def test_box_pk():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    
    kin = jnp.geomspace(1e-3, 1e1, 1000)
    pkin = pk(kin)
    pkout = mock_box(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.))
    
    ax = plt.gca()
    maskin = kin < pkout.edges[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    ax.plot(pkout.k, pkout.k * pkout.power, label='output')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'box_pk.png')


def test_box_wmat():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)

    pk_mock = jax.jit(lambda pkin: mock_box(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.)))(pkin)

    wmat = jax.jacrev(lambda pkin: mock_box(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.)).power)(kin**(-3))
    pk_theory = wmat.dot(pkin)
    
    ax = plt.gca()
    maskin = kin < pk_mock.edges[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    ax.plot(pk_mock.k, pk_mock.k * pk_mock.power, label='mock')
    ax.plot(pk_mock.k, pk_mock.k * pk_theory, label='wmat * input')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'box_wmat.png')


def gaussian_survey(boxsize=2000., meshsize=128, size=int(1e6), seed=42, scale=0.2, **kwargs):
    from jax import random
    if isinstance(seed, int):
        seed = random.key(seed)
    return SurveySelection(scale * boxsize * random.normal(seed, shape=(size, 3)), boxcenter=0., boxsize=boxsize, meshsize=meshsize, **kwargs)

    
def mock_survey(power, selection, unitary_amplitude=True, norm=None, seed=42, **kwargs):
    field = GaussianField(power, boxsize=selection.boxsize, meshsize=selection.meshsize, unitary_amplitude=unitary_amplitude, seed=seed, **kwargs)
    edges = {'step': 0.01}
    return SurveyFFTPower(field.density * selection.density, selection.density, cellsize=field.cellsize, edges=edges, compensate=None, norm=norm)


def test_survey_pk():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    
    kin = jnp.geomspace(1e-3, 1e1, 1000)
    pkin = pk(kin)
    selection = gaussian_survey()
    pkout = mock_survey(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), selection=selection)
    
    ax = plt.gca()
    maskin = kin < pkout.edges[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    ax.plot(pkout.k, pkout.k * pkout.power, label='output')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'survey_pk.png')


def test_survey_wmat():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)

    selection = gaussian_survey()
    pk_mock = jax.jit(lambda pkin: mock_survey(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), selection, norm=1.))(pkin)

    wmat = jax.jacrev(lambda pkin: mock_survey(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), selection, norm=1.).power)(pkin)
    pk_theory = wmat.dot(pkin)
    
    ax = plt.gca()
    maskin = kin < pk_mock.edges[-1]
    #ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    ax.plot(pk_mock.k, pk_mock.k * pk_mock.power, label='mock')
    ax.plot(pk_mock.k, pk_mock.k * pk_theory, label='wmat * input')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'survey_wmat.png')


def test_survey_wmat(npk=10, nwmat=10):
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)

    selection = gaussian_survey()
    get_pk = jax.jit(lambda seed: mock_survey(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), selection, seed=seed, unitary_amplitude=False))
    get_wmat = jax.jit(lambda seed: jax.jacrev(lambda pkin: mock_survey(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), selection, seed=seed, unitary_amplitude=False).power)(pkin))
    from tqdm import trange
    pks, pkts = [], []
    with trange(npk) as t:
        for imock in t:
            seed = random.key(2 * imock)
            pks.append(get_pk(seed))
    with trange(nwmat) as t:
        for imock in t:
            seed = random.key(2 * imock + 1)
            pkts.append(get_wmat(seed).dot(pkin))
    k, edges = pks[0].k, pks[0].edges
    pks = [pk.power for pk in pks]
    pk_mean, pk_std = np.mean(pks, axis=0), np.std(pks, axis=0) / npk**0.5
    pkt_mean, pkt_std = np.mean(pkts, axis=0), np.std(pkts, axis=0) / nwmat**0.5
    ax = plt.gca()
    maskin = kin < edges[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    ax.fill_between(k, k * (pk_mean - pk_std), k * (pk_mean + pk_std), lw=0., alpha=0.5, label='mocks')
    ax.fill_between(k, k * (pkt_mean - pkt_std), k * (pkt_mean + pkt_std), lw=0., alpha=0.5, label='wmat * input')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'survey_wmat.png')
    

if __name__ == '__main__':
    
    #test_box_pk()
    #test_box_wmat()
    #test_survey_pk()
    test_survey_wmat(npk=100, nwmat=100)