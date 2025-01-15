from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp
from jax import random

from matplotlib import pyplot as plt

from jaxpower import generate_gaussian_mesh, ParticleField, FKPField, compute_mesh_power, compute_fkp_power, compute_normalization, utils


dirname = Path('_tests')


def mock_box(power, unitary_amplitude=True, boxsize=1000., meshsize=128, seed=42):
    mesh = generate_gaussian_mesh(lambda kvec: power(sum(kk**2 for kk in kvec)**0.5), unitary_amplitude=unitary_amplitude, boxsize=boxsize, meshsize=meshsize, seed=seed)
    edges = {'step': 0.01}
    return compute_mesh_power(mesh, edges=edges, los='x')


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
    ax.plot(pkout.k, pkout.k * pkout.power[0].real, label='output')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'box_pk.png')


def test_box_wmat():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)
    
    get_pk = lambda pkin, **kwargs: mock_box(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), **kwargs)
    get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).power[0].real)(pkin)
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
    ax.plot(k, k * pk.power[0].real, label='$P_o(k)$')
    ax.plot(k, k * pkt, label='$dP_o/dP_i \cdot P_i$')
    ax.plot(k, k * pkt_km3, label='$dP_o/dP_i | k^{-3} \cdot P_i$')
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    ax.set_ylabel(r'$k P(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'box_wmat.png')


def gaussian_survey(boxsize=2000., meshsize=128, boxcenter=0., size=int(1e6), seed=random.key(42), scale=0.03, paint=False):
    # Generate Gaussian-distributed positions
    positions = scale * boxsize * random.normal(seed, shape=(size, 3))
    toret = ParticleField(positions + boxcenter, boxcenter=boxcenter, boxsize=boxsize, meshsize=meshsize)
    if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
    return toret


def mock_survey(power, selection, unitary_amplitude=True, seed=random.key(42), **kwargs):
    # Generate Gaussian field
    mesh = generate_gaussian_mesh(lambda kvec: power(sum(kk**2 for kk in kvec)**0.5), boxsize=selection.boxsize, meshsize=selection.meshsize, unitary_amplitude=unitary_amplitude, seed=seed, **kwargs)
    edges = {'step': 0.01}
    # Multiply Gaussian field with survey selection function, then compute power spectrum
    norm = compute_normalization(selection, selection)
    return compute_mesh_power(mesh * selection, edges=edges).clone(norm=norm)


def mock_survey_noise(power, boxsize=2000., meshsize=128, size=int(1e6), seed=random.key(42), unitary_amplitude=True):
    # Generate Gaussian field
    seeds = random.split(seed, 4)
    # randoms
    randoms = gaussian_survey(boxsize=boxsize, meshsize=meshsize, size=size, seed=seeds[0], paint=False)
    # data
    data = gaussian_survey(boxsize=boxsize, meshsize=meshsize, size=size, seed=seeds[1], paint=False)
    # field
    mesh = generate_gaussian_mesh(lambda kvec: power(sum(kk**2 for kk in kvec)**0.5),
                                  boxsize=randoms.boxsize, meshsize=randoms.meshsize,
                                  boxcenter=0., unitary_amplitude=unitary_amplitude, seed=seeds[2])
    # weights = 1 + delta
    data = data.clone(weights=data.weights * (1. + mesh.read(data.positions, resampler='cic', compensate=True)))
    edges = {'step': 0.01}
    fkp = FKPField(data, randoms)
    power = compute_fkp_power(fkp, edges=edges, resampler='tsc', interlacing=3)
    return power

def test_survey_pk():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    
    kin = jnp.geomspace(1e-3, 1e1, 1000)
    pkin = pk(kin)
    selection = gaussian_survey(paint=True)
    #pkout = mock_survey(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), selection=selection)
    pkout = mock_survey_noise(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.))
    
    ax = plt.gca()
    maskin = kin < pkout.edges[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    ax.plot(pkout.k, pkout.k * pkout.power[0].real, label='output')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'survey_pk.png')


def test_survey_wmat(npk=10, npkt=10):
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)
    selection = gaussian_survey(paint=True)
    get_pk = lambda pkin, **kwargs: mock_survey(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), selection, **kwargs)
    get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).power[0].real)(pkin)
    get_pk = jax.jit(get_pk, static_argnames=['unitary_amplitude'])
    get_wmat = jax.jit(get_wmat, static_argnames=['unitary_amplitude'])

    from tqdm import trange
    pks, pkts, pkts_km3 = [], [], []

    with trange(npk) as t:
        for imock in t:
            seed = random.key(2 * imock)
            pks.append(get_pk(pkin, seed=seed))
    with trange(npkt) as t:
        for imock in t:
            # Here we do not use the same seeds
            seed = random.key(2 * imock + 1)
            wmat = get_wmat(pkin, seed=seed)
            wmat_km3 = get_wmat(kin**(-3), seed=seed)
            pkts.append(wmat.dot(pkin))
            pkts_km3.append(wmat_km3.dot(pkin))

    k, edges = pks[0].k, pks[0].edges
    pk_mean, pk_std = np.mean([pk.power[0].real for pk in pks], axis=0), np.std([pk.power[0].real for pk in pks], axis=0) / npk**0.5
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
    

def test_survey_wmat_noise(npk=10, npkt=10):
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)
    get_pk = lambda pkin, **kwargs: mock_survey_noise(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), **kwargs)
    get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).power[0].real)(pkin)

    from tqdm import trange
    pks, pkts, pkts_km3 = [], [], []

    with trange(npk) as t:
        for imock in t:
            seed = random.key(2 * imock)
            pks.append(get_pk(pkin, seed=seed))
    with trange(npkt) as t:
        for imock in t:
            # Here we do not use the same seeds
            seed = random.key(2 * imock + 1)
            wmat = get_wmat(pkin, seed=seed)
            wmat_km3 = get_wmat(kin**(-3), seed=seed)
            pkts.append(wmat.dot(pkin))
            pkts_km3.append(wmat_km3.dot(pkin))

    k, edges = pks[0].k, pks[0].edges
    pk_mean, pk_std = np.mean([pk.power[0].real for pk in pks], axis=0), np.std([pk.power[0].real for pk in pks], axis=0) / npk**0.5
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
    utils.savefig(dirname / 'survey_wmat_noise.png')


if __name__ == '__main__':

    test_survey_pk()
    exit()
    
    test_box_pk()
    test_box_wmat()
    test_survey_pk()
    test_survey_wmat(npk=10, npkt=10)
    test_survey_wmat_noise(npk=10, npkt=10)