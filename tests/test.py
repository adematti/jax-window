import os
from pathlib import Path
from functools import partial

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.' # NOTE: jax preallocates GPU (default 75%)

import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt

from jaxpower import generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, ParticleField, FKPField, compute_mesh_power, compute_fkp_power, compute_normalization, PowerSpectrumMultipoles, utils

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
    maskin = kin < pkout.edges(projs=0)[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    ax.plot(pkout.x(projs=0), pkout.x(projs=0) * pkout.view(projs=0).real, label='output')
    ax.legend(frameon=False)
    utils.savefig(dirname / 'box_pk.png')


def test_box_wmat():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)

    get_pk = lambda pkin, **kwargs: mock_box(lambda k: jnp.interp(k, kin, pkin, left=0., right=0.), **kwargs)
    get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).view(projs=0).real)(pkin)
    get_pk = jax.jit(get_pk, static_argnames=['unitary_amplitude'])
    get_wmat = jax.jit(get_wmat, static_argnames=['unitary_amplitude'])

    seed = random.key(42)
    # Get one power spectrum
    pk = get_pk(pkin, seed=seed)
    pkt = get_wmat(pkin, seed=seed).dot(pkin)
    pkt_km3 = get_wmat(kin**(-3), seed=seed).dot(pkin)
    k, edges = pk.x(projs=0), pk.edges(projs=0)

    ax = plt.gca()
    maskin = kin < edges[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input $P_i$')
    ax.plot(k, k * pk.view(projs=0).real, label='$P_o(k)$')
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
    maskin = kin < pkout.edges(projs=0)[-1]
    ax.plot(kin[maskin], kin[maskin] * pkin[maskin], label='input')
    ax.plot(pkout.x(projs=0), pkout.x(projs=0) * pkout.view(projs=0).real, label='output')
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
    get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).view(projs=0).real)(pkin)
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

    k, edges = pks[0].x(projs=0), pks[0].edges(projs=0)
    pk_mean, pk_std = np.mean([pk.view(projs=0).real for pk in pks], axis=0), np.std([pk.view(projs=0).real for pk in pks], axis=0) / npk**0.5
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
    get_wmat = lambda pkin, **kwargs: jax.jacrev(lambda pkin: get_pk(pkin, **kwargs).view(projs=0).real)(pkin)

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

    k, edges = pks[0].x(projs=0), pks[0].edges(projs=0)
    pk_mean, pk_std = np.mean([pk.view(projs=0).real for pk in pks], axis=0), np.std([pk.view(projs=0).real for pk in pks], axis=0) / npk**0.5
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


def test_misc():
    from jaxwindow.mock import wigner3j_square

    for ell1, ell2 in [(2, 2), (2, 4), (4, 4)]:
        print(ell1, ell2, wigner3j_square(ell1, ell2, prefactor=False))


def test_anisotropic(npk=10):

    #from jax import config
    #config.update('jax_enable_x64', True)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='eisenstein_hu_nowiggle')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin) * (1. + 0.2 * np.sin(kin / 0.006))
    ells = (0, 2, 4)

    f, b = 0.9, 1.5
    pkb = b**2 * pkin
    beta = f / b
    poles = [(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pkb,
              0.8 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pkb,
              8. / 35 * beta ** 2 * pkb]

    def make_callable(poles):
        toret = {}
        def get_fun(ill):
            return lambda k: jnp.interp(k, kin, poles[ill], left=0., right=0.)
        for ill, ell in enumerate(ells):
            toret[ell] = get_fun(ill)
        return toret

    @partial(jax.jit, static_argnames=['los'])
    def mock(seed, los='x', unitary_amplitude=True):
        attrs = dict(boxsize=1000., boxcenter=(1e9, 0., 0.), meshsize=64)
        mesh = generate_anisotropic_gaussian_mesh(make_callable(poles), los=los, seed=seed, unitary_amplitude=unitary_amplitude, **attrs)
        edges = {'step': 0.01}
        return compute_mesh_power(mesh, edges=edges, los={'local': 'firstpoint'}.get(los, los), ells=ells)

    list_los = ['x', 'local']
    list_color = ['C0', 'C1']

    ax = plt.gca()
    ax.plot([], [], color='k', label='input')
    for los, color in zip(list_los, list_color):
        pks = [mock(random.key(i * 42), los=los) for i in range(npk)]
        power = pks[0]
        pk_mean, pk_std = np.mean([power.view().real for power in pks], axis=0), np.std([power.view().real for power in pks], axis=0) / npk**0.5
        ax.plot([], [], color=color, label=los)
        for ill, ell in enumerate(ells):
            ax.fill_between(power.x(projs=ell), power.x(projs=ell) * (pk_mean - pk_std)[ill], power.x(projs=ell) * (pk_mean + pk_std)[ill], color=color, lw=0.5, alpha=0.8)
    maskin = (kin >= power.edges(projs=0)[0]) & (kin <= power.edges(projs=0)[-1])
    for ill, ell in enumerate(ells):
        ax.plot(kin[maskin], kin[maskin] * poles[ill][maskin], color='k', linestyle='--')
    ax.legend()
    plt.show()


def test_misc():
    from tqdm import trange
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin)
    boxsize, meshsize = 1000., 64
    selection = gaussian_survey(boxsize=boxsize, meshsize=meshsize, boxcenter=0., paint=True)

    def delta_w(delta_i):
        delta_w = delta_i.c2r()
        delta_w *= selection
        return delta_w.r2c()

    power = lambda k: jnp.interp(k, kin, pkin, left=0., right=0.)
    pkvec = lambda kvec: power(sum(kk**2 for kk in kvec)**0.5)

    if False:
        @jax.jit
        def get_pk(seed):
            mesh = generate_gaussian_mesh(pkvec, boxsize=selection.boxsize, meshsize=selection.meshsize, boxcenter=selection.boxcenter, seed=seed)
            edges = {'step': 0.01}
            return compute_mesh_power(mesh, edges=edges)

        get_pk(random.key(42))
        npk = 10
        with trange(npk) as t:
            for imock in t:
                get_pk(random.key(imock + 42))
    """
    @jax.jit
    def mock(pkin, seed):
        power = lambda k: jnp.interp(k, kin, pkin, left=0., right=0.)
        pkvec = lambda kvec: power(sum(kk**2 for kk in kvec)**0.5)
        mesh = generate_gaussian_mesh(pkvec, boxsize=selection.boxsize, meshsize=selection.meshsize, boxcenter=selection.boxcenter, seed=seed).r2c()
        P = mesh.clone(value=pkvec(mesh.coords(sparse=True)).astype(mesh.dtype) / mesh.cellsize.prod())
        power = jax.jvp(delta_w, (mesh,), (jax.jvp(delta_w, (mesh,), (P,))[1].conj(),))[1]
        edges = jnp.arange(0., np.min(np.pi / power.cellsize), 0.01)
        khat = power.coords(sparse=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in khat))
        ibin = jnp.digitize(knorm, edges)
        nmodes = jnp.full_like(power.value, 2, dtype='i4')
        nmodes = nmodes.at[..., 0].set(1)
        if power.shape[-1] % 2 == 0:
            nmodes = nmodes.at[..., -1].set(1)
        rdtype = power.real.dtype
        norm = jnp.prod(power.meshsize, dtype=rdtype) / jnp.prod(power.cellsize, dtype=rdtype)
        ibin, nmodes = ibin.ravel(), nmodes.ravel()
        power = power.ravel() * nmodes
        nmodes = jnp.bincount(ibin, weights=nmodes, length=len(edges) + 1)[1:-1]
        return jnp.bincount(ibin, weights=power, length=len(edges) + 1)[1:-1] / nmodes / norm

    wmat = mock(pkin, random.key(42))

    npk = 10
    with trange(npk) as t:
        for imock in t:
            wmat2 = mock(pkin, random.key(42 * imock))
            assert np.allclose(wmat2, wmat)
    """
    selectionk = selection.r2c()
    norm = compute_normalization(selection, selection)

    def delta_w(delta_i):
        # The forward model for the window product
        delta_w = delta_i.c2r()
        delta_w *= selection
        return delta_w.r2c()

    @jax.jit
    def mock_survey_average(pkin, seed):
        power = lambda k: jnp.interp(k, kin, pkin, left=0., right=0.)
        pkvec = lambda kvec: power(sum(kk**2 for kk in kvec)**0.5)
        P = selectionk.clone(value=pkvec(selectionk.coords(sparse=True)).astype(selectionk.dtype) / selectionk.cellsize.prod())
        #mesh = selectionk  # can be anything
        mesh = generate_gaussian_mesh(pkvec, boxsize=selection.boxsize, meshsize=selection.meshsize, boxcenter=selection.boxcenter, seed=seed).r2c()
        power = jax.jvp(delta_w, (mesh,), (jax.jvp(delta_w, (mesh,), (P,))[1].conj(),))[1]
        #power = mesh * mesh.conj()
        edges = np.arange(0., np.min(np.pi / power.cellsize), 0.005)
        kvec = power.coords(sparse=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        ibin = jnp.digitize(knorm, edges)
        nmodes = jnp.full_like(power.value, 2, dtype='i4')
        nmodes = nmodes.at[..., 0].set(1)
        if power.shape[-1] % 2 == 0:
            nmodes = nmodes.at[..., -1].set(1)
        rdtype = power.real.dtype
        #norm = jnp.prod(power.meshsize, dtype=rdtype) / jnp.prod(power.cellsize, dtype=rdtype)
        ibin, nmodes = ibin.ravel(), nmodes.ravel()
        knorm, power = knorm.ravel() * nmodes, power.ravel() * nmodes
        nmodes = jnp.bincount(ibin, weights=nmodes, length=len(edges) + 1)[1:-1]
        knorm = jnp.bincount(ibin, weights=knorm, length=len(edges) + 1)[1:-1] / nmodes
        power_nonorm = jnp.bincount(ibin, weights=power, length=len(edges) + 1)[1:-1] / nmodes
        return power_nonorm

    pk0 = mock_survey_average(pkin, random.key(42))

    npk = 10
    with trange(npk) as t:
        for imock in t:
            pk = mock_survey_average(pkin, random.key(42 * imock))
            assert np.allclose(pk, pk0)


def test_bspline():
    from matplotlib import pyplot as plt
    from scipy import interpolate
    xp = np.linspace(0., 0.2, 100)
    yp = np.linspace(0., 0.2, 300)

    def func(x, y):
        sigma = 0.2 * np.std(yp)
        return (1. + x**2) * np.exp(-(y - x)**2 / 2 / sigma)

    xx, yy, zz = [], [], []
    for x in xp[::3]:
        for y in yp:
            xx.append(x)
            yy.append(y - x)
            zz.append(func(x, y))
    tck = interpolate.bisplrep(xx, yy, zz)
    #tx = xp[::10]
    #ty = yp[1:-3:10] - xp[len(xp) // 2]
    #spline = interpolate.LSQBivariateSpline(x=xx, y=yy, z=zz, tx=tx, ty=ty)
    xe, ye = np.meshgrid(xp, yp, indexing='ij')
    fig, lax = plt.subplots(nrows=1, ncols=2, sharey=True)
    lax[0].pcolormesh(xp, yp, func(xe, ye).T)
    #lax[1].pcolormesh(xp, yp, spline(xe, ye, grid=False).T)
    tmp = np.array([interpolate.bisplev(xx, yp - xx, tck) for xx in xp])
    lax[1].pcolormesh(xp, yp, tmp.T)
    plt.show()


def test_window_matrix_estimator():
    from jaxpower import compute_mean_mesh_power, BinnedStatistic
    from jaxwindow import WindowMatrixEstimator
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu_nowiggle')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    kin = jnp.geomspace(1e-3, 1e1, 200)
    pkin = pk(kin) * (1. + 0.2 * np.sin(kin / 0.006))
    ells = (0, 2, 4)

    f, b = 0.9, 1.5
    pkb = b**2 * pkin
    beta = f / b
    poles = [(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pkb,
              0.8 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pkb,
              8. / 35 * beta ** 2 * pkb]
    theory = BinnedStatistic(x=(kin,) * len(ells), value=poles, projs=ells)

    boxsize, meshsize = 1000., 32
    edges = {'step': 0.01}
    selection = gaussian_survey(boxsize=boxsize, meshsize=meshsize, boxcenter=0., paint=True)
    mod = 1. + 0.01 * (random.uniform(random.key(42), shape=selection.shape) - 0.5)  # modulation

    def apply_selection(mesh, selection, cv=False):
        if cv:
            return mesh * selection
        return mesh * selection * mod

    def make_callable(theory):
        def get_fun(proj): return lambda x: jnp.interp(x, theory.x(projs=proj), theory.view(projs=proj), left=0., right=0.)
        return {proj: get_fun(proj) for proj in theory.projs}

    def mock_mean(theory, selection, los='x'):
        return compute_mean_mesh_power(selection, theory=make_callable(theory), edges=edges, los=los, ells=ells)

    def mock_diff(theory, selection, los='x', seed=42, unitary_amplitude=True):
        mesh = generate_anisotropic_gaussian_mesh(make_callable(theory), los=los, seed=seed,
                                                  unitary_amplitude=unitary_amplitude, **selection.attrs)
        toret = [compute_mesh_power(apply_selection(mesh, selection, cv=cv), edges=edges, los=los, ells=ells) for cv in [False, True]]
        return toret[0].clone(value=toret[0].view() - toret[1].view())

    estimator = WindowMatrixEstimator(theory=theory)
    #estimator.cv(mock_mean, func_kwargs=dict(selection=selection), indices=(Ellipsis, list(range(5))))
    estimator.sample(mock_diff, nmocks=3, func_kwargs=dict(selection=selection))
    #estimator.sample(mock_diff, func_kwargs=dict(selection=selection), indices=(Ellipsis, list(range(5))))
    wmat = estimator.mean(interp=False)
    wmat = estimator.mean(interp=True)
    std = estimator.std()


if __name__ == '__main__':

    test_box_pk()
    test_box_wmat()
    test_survey_pk()
    test_survey_wmat(npk=10, npkt=10)
    test_survey_wmat_noise(npk=10, npkt=10)
    test_anisotropic(npk=10)
    test_window_matrix_estimator()