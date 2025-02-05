import os
from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp
from jax import random

from cosmoprimo.fiducial import DESI
from jaxpower import (RealMeshField, ParticleField, FKPField,
compute_mesh_power, compute_fkp_power, compute_normalization, setup_logging, utils)
from jaxwindow.mock import generate_anisotropic_gaussian_mesh


ells = (0, 2, 4)
region = 'NGC'
cosmo = DESI()

data_dir = Path(os.getenv('SCRATCH')) / 'jax-window' / 'bgs'
plot_dir = Path('_tests')


def get_fn(base, imock=None):
    basename = '{base}'
    if imock is not None:
        basename += '_{imock:d}'
    return str(data_dir / (basename.format(base=base, imock=imock) + '.npy'))


def get_selection_particles():
    from mockfactory import Catalog, sky_to_cartesian
    base_dir = Path('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummitBGS/desipipe/v1/complete/2pt/merged/')
    positions, weights = [], []
    for iran in range(5):
        fn = base_dir / f'BGS_BRIGHT-21.5_{region}_{iran:d}_clustering.ran.fits'
        catalog = Catalog.read(fn)
        catalog[catalog.columns()]
        dist = cosmo.comoving_radial_distance(catalog['Z'])
        positions.append(sky_to_cartesian(dist, catalog['RA'], catalog['DEC']))
        weights.append(catalog['WEIGHT'] * catalog['WEIGHT_FKP'])
    positions, weights = np.concatenate(positions, axis=0), np.concatenate(weights, axis=0)
    particles = ParticleField(positions, weights=weights, cellsize=6., boxsize=4000.)
    return particles


def save_selection_mesh():
    selection = get_selection_particles()
    mesh = selection.paint(resampler='tsc', interlacing=3, compensate=True)
    mesh.save(get_fn(base='selection_mesh'))


def get_desi_power(selection='box', klim=(0., np.inf, 0.001), ells=(0, 2, 4)):
    from pypower import PowerSpectrumMultipoles
    nmocks = 25
    powers = []
    if selection == 'survey':
        for imock in range(nmocks):
            fn = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummitBGS/desipipe/v1/complete/baseline_2pt/mock{imock:d}/pk/pkpoles_BGS_BRIGHT-21.5_{region}_z0.1-0.4.npy'
            power = PowerSpectrumMultipoles.load(fn).select(klim)
            k = power.k
            powers.append(power(ell=ells, complex=False))
    if selection == 'box':
        for imock in range(nmocks):
            fn = f'/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/desipipe/BGS_v0.1/2pt/mock{imock:d}_los-x/pk/pkpoles_BGS_z0.2000_lin_cellsize2_boxsize2000.npy'
            power = PowerSpectrumMultipoles.load(fn).select(klim)
            k = power.k
            powers.append(power(ell=ells, complex=False))
    return k, np.mean(powers, axis=0), np.std(powers, axis=0) / nmocks**0.5


def fit_mock_power(klim=(0.02, 0.1, 0.005), plot=False):
    pk_cosmo = cosmo.get_fourier().pk_interpolator(of='delta_cb').to_1d(z=0.)
    ko, mean, std = get_desi_power(selection='box', klim=klim, ells=ells)
    ratio0 = mean[0][-1] / pk_cosmo(ko[-1])
    power0 = ratio0 * pk_cosmo(ko)

    from jaxwindow.mock import generate_acceptable_poles

    def func(ko, alpha1, alpha2):
        return np.ravel(generate_acceptable_poles(power0, alpha1=alpha1, alpha2=alpha2))

    from scipy.optimize import curve_fit
    popt = curve_fit(func, ko, mean.ravel(), p0=[0.1, 1.], sigma=std.ravel())[0]
    popt[1] = min(popt[1], 0.99)  # < 1. to guarantee bijection

    kin = jnp.geomspace(1e-3, 1e1, 200)
    poles = jnp.array(generate_acceptable_poles(ratio0 * pk_cosmo(kin), alpha1=popt[0], alpha2=popt[1]))

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        maskin = (kin >= klim[0]) & (kin < klim[1])
        ax.plot([], [], color='C0', label='box power')
        ax.plot([], [], color='C1', label='"fitted" power')
        for ill, ell in enumerate(ells):
            ax.plot(ko, ko * mean[ill], color='C0')
            ax.plot(kin[maskin], kin[maskin] * poles[ill, maskin], color='C1')
        ax.legend(frameon=False)
        utils.savefig(plot_dir / 'poles.png')
    return kin, poles


def generate_gaussian_mesh_mocks(nmocks=100):
    selection = RealMeshField.load(get_fn(base='selection_mesh'))
    kin, poles = fit_mock_power()
    np.save(get_fn(base='pkin'), {'k': kin, 'power': poles})

    def mock(poles, selection, unitary_amplitude=True, seed=random.key(42)):
        # Generate Gaussian field
        mesh = generate_anisotropic_gaussian_mesh(kin, poles, unitary_amplitude=unitary_amplitude, boxsize=selection.boxsize, meshsize=selection.meshsize, boxcenter=selection.boxcenter, los='local', seed=seed)
        edges = {'step': 0.01}
        # Multiply Gaussian field with survey selection function, then compute power spectrum
        norm = compute_normalization(selection, selection)
        return compute_mesh_power(mesh * selection, edges=edges, ells=ells, los='firstpoint').clone(norm=norm)

    def mock(p, unitary_amplitude=True, seed=random.key(42)):
        # Generate Gaussian field
        p = poles.at[0, 10:11].set(p)
        mesh = generate_anisotropic_gaussian_mesh(kin, p, unitary_amplitude=unitary_amplitude, boxsize=selection.boxsize, meshsize=selection.meshsize, boxcenter=selection.boxcenter, los='local', seed=seed)
        edges = {'step': 0.01}
        norm = compute_normalization(selection, selection)
        #edges = jnp.arange(0., np.min(np.pi / mesh.cellsize), 0.01)
        # Multiply Gaussian field with survey selection function, then compute power spectrum
        #khat = mesh.coords(sparse=True)
        #mesh = mesh.real**2 + mesh.imag**2
        #knorm = jnp.sqrt(sum(kk**2 for kk in khat))
        #index = jnp.digitize(knorm, edges)
        #return jnp.bincount(index.ravel(), weights=mesh.ravel(), length=len(edges) + 1)
        return compute_mesh_power(mesh * selection, edges=edges, ells=ells, los='firstpoint').clone(norm=norm)

    get_pk = lambda pkin, **kwargs: mock(pkin, **kwargs)
    #get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).view())(pkin)
    get_wmat = lambda pkin, **kwargs: jax.jacfwd(lambda pkin: get_pk(pkin, **kwargs))(pkin)
    get_pk = jax.jit(get_pk)
    get_wmat = jax.jit(get_wmat)

    from tqdm import trange
    with trange(nmocks) as t:
        for imock in t:
            seed = random.key(2 * imock)
            pk = get_pk(poles[0, 10:11], seed=seed)
            jax.block_until_ready(pk)
            #pk.save(get_fn(base='pk', imock=imock))
            # Here we do not use the same seeds
            seed = random.key(2 * imock + 1)
            wmat = get_wmat(poles[0, 10:11], seed=seed)
            jax.block_until_ready(wmat)
            #np.save(get_fn(base='wmat', imock=imock), wmat)


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.' # NOTE: jax preallocates GPU (default 75%)

    setup_logging()
    utils.mkdir(data_dir)
    #save_selection_mesh()
    #fit_mock_power(plot=True)
    generate_gaussian_mesh_mocks(nmocks=100)
