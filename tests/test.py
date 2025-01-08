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
    
    get_pk = lambda pkin, **kwargs: mock_box(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), **kwargs)
    get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).power)(pkin)
    get_pk = jax.jit(get_pk, static_argnames=['unitary_amplitude'])
    get_wmat = jax.jit(get_wmat, static_argnames=['unitary_amplitude'])

    seed = random.key(42)
    # Get one power spectrum
    pk = get_pk(pkin, seed=seed)
    pkt = get_wmat(pkin, seed=seed).dot(pkin)
    pkt_km3 = get_wmat(kin**(-3), seed=seed).dot(pkin)
    k, edges = pk.k, pk.edges

    ax = plt.gca()
    maskin = kin < edges[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input $P_i$')
    ax.plot(k, k * pk.power, label='$P_o(k)$')
    ax.plot(k, k * pkt, label='$dP_o/dP_i \cdot P_i$')
    ax.plot(k, k * pkt_km3, label='$dP_o/dP_i | k^{-3} \cdot P_i$')
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    ax.set_ylabel(r'$k P(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'box_wmat.png')


def get_gaussian_survey_positions(boxsize=2000., meshsize=128, size=int(1e6), seed=random.key(42), scale=0.2):
    # Generate Gaussian-distributed positions
    positions = scale * boxsize * random.normal(seed, shape=(size, 3))
    # weights = 0. outside of the box extent
    mask = jnp.all((positions >= -boxsize / 2.) & (positions <= boxsize / 2.), axis=-1)
    weights = jnp.where(mask, 1., 0.)
    return positions, weights


def gaussian_survey(boxsize=2000., meshsize=128, size=int(1e6), seed=random.key(42), scale=0.2, **kwargs):
    positions, weights = get_gaussian_survey_positions(boxsize=boxsize, meshsize=meshsize, size=size, seed=seed, scale=scale)
    # Survey selection function, obtained by painting the above positions
    return SurveySelection(positions, weights=weights, boxcenter=0., boxsize=boxsize, meshsize=meshsize, **kwargs)


def mock_survey(power, selection, unitary_amplitude=True, norm=None, seed=random.key(42), **kwargs):
    # Generate Gaussian field
    field = GaussianField(power, boxsize=selection.boxsize, meshsize=selection.meshsize,
                          unitary_amplitude=unitary_amplitude, seed=seed, **kwargs)
    edges = {'step': 0.01}
    # Multiply Gaussian field with survey selection function, then compute power spectrum
    return SurveyFFTPower(field.density * selection.density, selection.density, cellsize=field.cellsize, edges=edges, compensate=None, norm=norm)


def mock_survey_noise(power, boxsize=2000., meshsize=128, size=int(1e6), seed=random.key(42), scale=0.2, unitary_amplitude=True, norm=None):
    # Generate Gaussian field
    seeds = random.split(seed, 3)
    shotnoise_nonorm = 0.
    # randoms
    positions, weights = get_gaussian_survey_positions(boxsize=boxsize, meshsize=meshsize, size=size, seed=seeds[0], scale=scale)
    shotnoise_nonorm += jnp.sum(weights**2)
    randoms = SurveySelection(positions, weights, boxcenter=0., boxsize=boxsize, meshsize=meshsize)
    # field
    field = GaussianField(power, boxsize=randoms.boxsize, meshsize=randoms.meshsize,
                          unitary_amplitude=unitary_amplitude, seed=seeds[1])
    positions, weights = get_gaussian_survey_positions(boxsize=boxsize, meshsize=meshsize, size=size, seed=seeds[2], scale=scale)
    # data
    shotnoise_nonorm += jnp.sum(weights**2)
    weights *= field.read(positions)  # multiply by 1 + delta
    data = SurveySelection(positions, weights, boxcenter=0., boxsize=boxsize, meshsize=meshsize)
    edges = {'step': 0.01}
    # Multiply Gaussian field with survey selection function, then compute power spectrum
    return SurveyFFTPower(data.density, randoms.density, cellsize=field.cellsize, edges=edges, compensate='cic', norm=norm, shotnoise_nonorm=shotnoise_nonorm)


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


def test_survey_wmat(npk=10, npkt=10, noise=False):
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)
    if noise:
        get_pk = lambda pkin, **kwargs: mock_survey_noise(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), **kwargs)
    else:
        selection = gaussian_survey()
        get_pk = lambda pkin, **kwargs: mock_survey(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), selection, **kwargs)
    get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).power)(pkin)
    get_pk = jax.jit(get_pk, static_argnames=['unitary_amplitude'])
    get_wmat = jax.jit(get_wmat, static_argnames=['unitary_amplitude'])

    from tqdm import trange
    pks, wmats, pkts, pkts_km3 = [], [], [], []

    with trange(npk) as t:
        for imock in t:
            seed = random.key(2 * imock)
            pks.append(get_pk(pkin, seed=seed))

    with trange(npkt) as t:
        for imock in t:
            # Here we do not use the same seeds
            seed = random.key(2 * imock + 1)
            wmat = get_wmat(pkin, seed=seed)
            wmats.append(wmat)
            pkts.append(wmat.dot(pkin))
            pkts_km3.append(get_wmat(kin**(-3), seed=seed).dot(pkin))

    k, edges = pks[0].k, pks[0].edges
    pk_mean, pk_std = np.mean([pk.power for pk in pks], axis=0), np.std([pk.power for pk in pks], axis=0) / npk**0.5
    pkt_mean, pkt_std = np.mean(pkts, axis=0), np.std(pkts, axis=0) / npkt**0.5
    pkt_km3_mean, pkt_km3_std = np.mean(pkts_km3, axis=0), np.std(pkts_km3, axis=0) / npkt**0.5

    ax = plt.gca()
    maskin = kin < edges[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    kw = dict(lw=0., alpha=0.5)
    ax.fill_between(k, k * (pk_mean - pk_std), k * (pk_mean + pk_std), label=r'$\langle P_o(k) \rangle$', **kw)
    ax.fill_between(k, k * (pkt_mean - pkt_std), k * (pkt_mean + pkt_std), label=r'$\langle dP_o/dP_i \rangle \cdot P_i$', **kw)
    ax.fill_between(k, k * (pkt_km3_mean - pkt_std), k * (pkt_km3_mean + pkt_km3_std), lw=0., alpha=0.5, label=r'$\langle dP_o/dP_i | k^{-3} \rangle \cdot P_i$')
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    ax.set_ylabel(r'$k P(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'survey_wmat.png')
    

if __name__ == '__main__':
    
    #test_box_pk()
    #test_box_wmat()
    #test_survey_pk()
    #test_survey_wmat(npk=10, npkt=10)
    test_survey_wmat(npk=100, npkt=100, noise=True)