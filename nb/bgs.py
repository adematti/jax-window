import os
from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp
from jax import random

from cosmoprimo.fiducial import DESI
from jaxpower import (RealMeshField, ParticleField, FKPField,
compute_mesh_power, compute_fkp_power, generate_anisotropic_gaussian_mesh, compute_normalization, setup_logging, utils)
from jaxpower.utils import MemoryMonitor


ells = (0, 2, 4)
region = 'NGC'
cosmo = DESI()

data_dir = Path(os.getenv('SCRATCH')) / 'jax-window' / 'bgs'
plot_dir = Path('_tests_bgs')


def get_fn(base, imock=None, ext='npy'):
    basename = '{base}'
    if imock is not None:
        basename += '_{imock:d}'
    return str(data_dir / (basename.format(base=base, imock=imock) + '.' + ext))


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
    particles = ParticleField(positions, weights=weights, meshsize=400, boxsize=4000.)
    return particles


def save_selection_mesh():
    selection = get_selection_particles()
    mesh = selection.paint(resampler='tsc', interlacing=3, compensate=True)
    mesh.save(get_fn(base='selection_mesh', ext='npz'))


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

    from jaxpower import BinnedStatistic
    from jaxwindow.mock import generate_acceptable_poles

    def func(ko, alpha1, alpha2):
        return np.ravel(generate_acceptable_poles(power0, alpha1=alpha1, alpha2=alpha2))

    from scipy.optimize import curve_fit
    popt = curve_fit(func, ko, mean.ravel(), p0=[0.1, 1.], sigma=std.ravel())[0]
    popt[1] = min(popt[1], 0.99)  # < 1. to guarantee bijection

    kin = jnp.geomspace(1e-4, 0.5, 500)
    poles = jnp.array(generate_acceptable_poles(ratio0 * pk_cosmo(kin), alpha1=popt[0], alpha2=popt[1]))
    poles = BinnedStatistic(x=(kin,) * len(ells), value=tuple(poles), projs=ells)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        spoles = poles.select(xlim=klim)
        ax.plot([], [], color='C0', label='box power')
        ax.plot([], [], color='C1', label='"fitted" power')
        for ill, ell in enumerate(ells):
            ax.plot(ko, ko * mean[ill], color='C0')
            ki = spoles.x(projs=ell)
            ax.plot(ki, ki * spoles.view(projs=ell), color='C1')
        ax.legend(frameon=False)
        utils.savefig(plot_dir / 'poles.png')
    return poles


def estimate_window():
    from functools import partial
    from jaxpower import compute_mesh_power, compute_mesh_window, compute_mean_mesh_power, compute_normalization
    from jaxwindow import WindowMatrixEstimator

    selection = RealMeshField.load(get_fn(base='selection_mesh', ext='npz'))
    theory = fit_mock_power()
    theory.save(get_fn(base='theory'))
    edges = {'step': 0.005}
    los = 'local'
    ells = (0, 2, 4)
    norm = compute_normalization(selection, selection)

    # Apply selection function
    def apply_selection(mesh, selection, cv=False):
        # Selection function
        mesh = mesh * selection
        if not cv:  # with radial integral constraint
            dmin = np.min(selection.boxcenter - selection.boxsize / 2.)
            dmax = (1. + 1e-9) * np.sqrt(np.sum((selection.boxcenter + selection.boxsize / 2.)**2))
            edges = jnp.linspace(dmin, dmax, 1000)
            rnorm = jnp.sqrt(sum(xx**2 for xx in selection.coords(sparse=True))).ravel()
            ibin = jnp.digitize(rnorm, edges, right=False)
            bw = jnp.bincount(ibin, weights=mesh.ravel(), length=len(edges) + 1)
            b = jnp.bincount(ibin, weights=selection.ravel(), length=len(edges) + 1)
            # Integral constraint
            bw = bw / jnp.where(b == 0., 1., b)  # (integral of W * delta) / (integral of W)
            mesh -= bw[ibin].reshape(selection.shape) * selection
        return mesh

    # Difference: mock - control variate power spectrum

    def mock_survey(theory, selection, with_cv=True, seed=42, unitary_amplitude=True):
        mesh = generate_anisotropic_gaussian_mesh(theory, los=los, seed=seed,
                                                  unitary_amplitude=unitary_amplitude, **selection.attrs)
        kw = dict(edges=edges, los={'local': 'firstpoint'}.get(los, los), ells=ells)
        power = compute_mesh_power(apply_selection(mesh, selection, cv=False), **kw).clone(norm=norm)
        if with_cv:
            cv = compute_mesh_power(apply_selection(mesh, selection, cv=True), **kw).clone(norm=norm)
            return power.clone(value=power.view() - cv.view())
        return power

    if False:
        def mock_survey(theory, selection, with_cv=True, seed=42, unitary_amplitude=False):
            mesh = generate_anisotropic_gaussian_mesh(theory, los=los, seed=seed,
                                                     unitary_amplitude=unitary_amplitude, **selection.attrs)
            mesh = mesh * selection
            mesh = mesh.r2c()
            return (mesh * mesh.conj()).sum()

    if True:
        get_pk = lambda pkin, *args, **kwargs: mock_survey(pkin, *args, with_cv=False, unitary_amplitude=False, **kwargs)
        get_pk = jax.jit(get_pk)

        from tqdm import trange
        pks_ic = []
        with MemoryMonitor() as mem:
            for imock in trange(4):
                seed = random.key(2 * imock + 1)
                pks_ic.append(get_pk(theory, selection, seed=seed))
                #pks_ic.append(get_pk(selection))
                jax.block_until_ready(pks_ic[-1])
                mem()
        #exit()

    estimator_cv_sparse = WindowMatrixEstimator(theory=theory)
    # Compute control variate
    estimator_cv_sparse.cv(partial(compute_mesh_window, edges=edges, los=los, ells=ells, pbar=True, buffer='_tmp', norm=norm))
    # Sample
    nmocks = 25
    estimator_cv_sparse.sample(partial(mock_survey, with_cv=True), nmocks=nmocks,
                               func_kwargs=dict(selection=selection),
                               indices=(Ellipsis, slice(0, None, 4)))
    estimator_cv_sparse.sample(partial(mock_survey, with_cv=True), nmocks=2 * nmocks,
                               func_kwargs=dict(selection=selection),
                               indices=(Ellipsis, slice(10)))
    estimator_cv_sparse.sample(partial(mock_survey, with_cv=True), nmocks=nmocks,
                           func_kwargs=dict(selection=selection),
                           indices=(Ellipsis, slice(-1, None)))
    wmat_cv_sparse = estimator_cv_sparse.mean(interp=True)
    std_cv_sparse = estimator_cv_sparse.std(interp=True, std_on_mean=True)
    wmat_cv_sparse.save(get_fn(base='wmatrix'))
    std_cv_sparse.save(get_fn(base='wmatrix_std'))


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95' # NOTE: jax preallocates GPU (default 75%)

    setup_logging()
    utils.mkdir(data_dir)
    #save_selection_mesh()
    #fit_mock_power(plot=True)
    estimate_window()
