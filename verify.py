"""Smoke-test the compiled parts of a dev env.

    python verify.py [--env dev|dev-full|jaxgpu]

Each check imports a package and runs a tiny computation through its compiled
code.  Prints a pass/fail table; exits non-zero if anything failed.
"""
import argparse
import importlib
import importlib.util
import os
import sys
import time
import traceback

import numpy as np

RNG = np.random.default_rng(42)


def c_stderr(fn):
    """Run fn() and return (result, what C code wrote to fd 2)."""
    import tempfile
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            out = fn()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        return out, tmp.read().decode(errors="replace")


def corrfunc_theory():
    from Corrfunc.theory.DD import DD
    x, y, z = RNG.uniform(0, 100, (3, 2000))
    bins = np.linspace(1, 20, 6)
    res, err = c_stderr(lambda: DD(1, 2, bins, x, y, z, periodic=True, boxsize=100.0))
    assert res["npairs"].sum() > 0
    # A generic-x86-64 build (e.g. the conda-forge binary) warns on every call.
    if "but the compiler does not" in err:
        raise AssertionError("built without SIMD kernels: " + err.splitlines()[0])
    return f"DD npairs={res['npairs'].sum()} (2 threads, SIMD kernels)"


def corrfunc_mocks():
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
    ra = RNG.uniform(0, 10, 1000)
    dec = RNG.uniform(-5, 5, 1000)
    res = DDtheta_mocks(1, 2, np.logspace(-1, 0, 5), ra, dec)
    return f"DDtheta npairs={res['npairs'].sum()}"


def pyfnntw_query():
    import pyfnntw
    from scipy.spatial import cKDTree
    data = RNG.uniform(0, 1, (5000, 3))
    query = RNG.uniform(0, 1, (500, 3))
    d, _ = pyfnntw.Treef64(data, leafsize=32).query(query, 4)
    ref, _ = cKDTree(data).query(query, k=4)
    if np.allclose(d, ref):
        return "matches cKDTree"
    if np.allclose(d, ref**2):
        raise AssertionError("returns SQUARED distances (sum_stat assumes Euclidean)")
    raise AssertionError("distances differ from cKDTree")


def dsigma_import():
    importlib.import_module("dsigma.precompute")
    import dsigma
    return dsigma.__version__


def classy_compute():
    from classy import Class
    c = Class()
    c.set({"output": "mPk", "P_k_max_1/Mpc": 1.0})
    c.compute()
    s8 = c.sigma8()
    c.struct_cleanup()
    return f"sigma8={s8:.4f}"


def camb_run():
    import camb
    p = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.965)
    return f"age={camb.get_background(p).get_derived_params()['age']:.2f} Gyr"


def treecorr_nn():
    import treecorr
    x, y = RNG.uniform(0, 10, (2, 2000))
    nn = treecorr.NNCorrelation(min_sep=0.1, max_sep=1, nbins=5)
    nn.process(treecorr.Catalog(x=x, y=y))
    return f"npairs={nn.npairs.sum():.0f}"


def healpy_map():
    import healpy as hp
    m = np.arange(hp.nside2npix(16), dtype=float)
    hp.anafast(m, lmax=10)
    return hp.__version__


def jax_jit():
    import jax
    import jax.numpy as jnp
    f = jax.jit(lambda x: jnp.sum(jnp.sin(x) ** 2))
    f(jnp.arange(10.0)).block_until_ready()
    return f"{jax.__version__} on {jax.devices()[0].platform}"


def jax_gpu():
    import jax
    import jax.numpy as jnp
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        raise AssertionError(f"no GPU device; jax sees {jax.devices()}")
    x = jax.device_put(jnp.ones((2048, 2048)), gpus[0])
    assert float((x @ x).block_until_ready()[0, 0]) == 2048.0
    return f"{jax.__version__} on {gpus[0].device_kind}"


def optax_adam():
    import jax
    import jax.numpy as jnp
    import optax
    opt = optax.adam(0.1)
    p = jnp.array([3.0, -2.0])
    state = opt.init(p)

    def loss(p):
        return jnp.sum(p**2)

    @jax.jit
    def step(p, state):
        updates, state = opt.update(jax.grad(loss)(p), state, p)
        return optax.apply_updates(p, updates), state

    for _ in range(100):
        p, state = step(p, state)
    assert float(loss(p)) < 1e-2
    return f"{optax.__version__}, 100 adam steps on {next(iter(p.devices())).platform}"


def camb_pinned():
    import camb
    if camb.__version__ != "1.6.6":
        raise AssertionError(f"camb {camb.__version__}, not 1.6.6 (the emu_pk 2.1 training truth)")
    p = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.965,
                        WantTransfer=True, kmax=2.0)
    return f"{camb.__version__}, sigma8={camb.get_results(p).get_sigma8_0():.4f}"


def pymangle_import():
    import pymangle
    return pymangle.__file__


def pyccl_sigma8():
    import pyccl
    cosmo = pyccl.CosmologyVanillaLCDM()
    return f"sigma8={pyccl.sigma8(cosmo):.4f}"


def pymaster_field():
    import healpy as hp
    import pymaster as nmt
    nside = 16
    nmt.NmtField(np.ones(hp.nside2npix(nside)), [RNG.normal(size=hp.nside2npix(nside))])
    return nmt.__version__ if hasattr(nmt, "__version__") else "ok"


def galsim_draw():
    import galsim
    im = galsim.Gaussian(sigma=1.0).drawImage(nx=16, ny=16, scale=0.2)
    return f"flux={im.array.sum():.3f}"


def torch_tensor():
    import torch
    return f"{torch.__version__}, sum={torch.arange(5.0).sum().item()}"


IMPORTS_DEV = [
    "numpy", "scipy", "astropy", "astropy_healpix", "h5py", "optax", "blackjax",
    "numpyro", "equinox", "numba", "getdist", "pixell", "emcee", "ultranest",
    "dustmaps", "extinction", "glass", "sklearn", "colossus", "cobaya",
    "cosmopower_jax", "jaxace", "CEmulator", "aemulusnu_hmf",
]
# own packages: skipped (not failed) when the repo is not cloned on this machine
OWN = [
    "emu_pk", "emu_hmf", "ggah_mod", "hod_mod", "sys_mapping", "xray_rr",
    "sum_stat", "ggah_cal", "rema", "ggah_bench",
]
IMPORTS_FULL = [
    "yt", "pyxsim", "soxs", "hdf5plugin", "diffrax", "dust_extinction",
    "astroquery", "specutils", "photutils", "halomod", "pyhalomodel",
    "jax_cosmo", "ppxf", "pyneb", "eazy", "discoeb",
    "pyhalomodel",  # installed --no-deps despite its numpy<2 pin
]
CHECKS_DEV = [
    corrfunc_theory, corrfunc_mocks, pyfnntw_query, dsigma_import,
    classy_compute, camb_run, treecorr_nn, healpy_map, jax_jit, pymangle_import,
]
CHECKS_FULL = [pyccl_sigma8, pymaster_field, galsim_draw, torch_tensor]
# jaxgpu: fails when jax does not see the GPU
IMPORTS_GPU = ["numpy", "scipy", "optax"]
OWN_GPU = ["emu_pk"]
CHECKS_GPU = [jax_gpu, optax_adam, camb_pinned, classy_compute]


def run(name, fn):
    t = time.perf_counter()
    try:
        info = fn()
        ok = True
    except Exception as e:  # noqa: BLE001 -- report everything
        info = f"{type(e).__name__}: {e}"
        if os.environ.get("VERIFY_TRACEBACK"):
            traceback.print_exc()
        ok = False
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {time.perf_counter() - t:5.1f}s  {info}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["dev", "dev-full", "jaxgpu"], default="dev")
    env = ap.parse_args().env
    print(f"python {sys.version.split()[0]} at {sys.prefix}")

    if env == "jaxgpu":
        imports, own, checks = IMPORTS_GPU, OWN_GPU, CHECKS_GPU
    else:
        full = env == "dev-full"
        imports = IMPORTS_DEV + (IMPORTS_FULL if full else [])
        own = OWN
        checks = CHECKS_DEV + (CHECKS_FULL if full else [])
    results = [run(m, lambda m=m: getattr(importlib.import_module(m), "__version__", "ok"))
               for m in imports]
    for m in own:
        if importlib.util.find_spec(m) is None:
            print(f"  SKIP  {m:<22}        not installed (repo not cloned?)")
        else:
            results.append(run(m, lambda m=m: getattr(importlib.import_module(m), "__version__", "ok")))
    results += [run(fn.__name__, fn) for fn in checks]

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
