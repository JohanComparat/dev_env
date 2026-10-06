# dev_env

Definition and creation of the conda environments used to develop every
project in `~/software` (not `~/software/previous/`).

| env        | what                                                                 |
|------------|----------------------------------------------------------------------|
| `dev`      | Python 3.12, CPU jax, all compiled deps, all own packages editable    |
| `dev-full` | `dev` + pyccl, NaMaster, galsim, yt/pyxsim, pytorch-cpu, notebook extras |
| `jaxgpu`   | Python 3.14, GPU jax (pip `jax[cuda13]`), optax, camb 1.6.6 + classy 3.3.4, emu_pk editable — GPU training, emu_pk 2.1 validation |

Conda is a Miniforge install at `~/software/miniforge3` (conda-forge only);
override with `CONDA_ROOT=/path/to/miniforge3 ./create.sh ...`.
Linux x86-64 only (the locks are `linux-64`).

## On a new machine

```bash
mkdir -p ~/software && cd ~/software
git clone git@github.com:JohanComparat/dev_env.git

# 1. Miniforge, into ~/software/miniforge3
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/software/miniforge3
~/software/miniforge3/bin/conda init bash

# 2. the own packages, side by side with dev_env (missing ones are skipped)
grep -vE '^\s*(#|$)' dev_env/own-packages.txt | while read -r r; do
  git clone git@github.com:JohanComparat/$r.git
done

# 3. the envs: --from-lock for the exact versions, or omit it to re-solve
cd dev_env
./create.sh dev --from-lock
./create.sh dev-full --from-lock
./create.sh jaxgpu --from-lock  # needs the NVIDIA driver (>= 580 for CUDA 13)
./system.sh                     # LaTeX + Julia (needs sudo)

# 4. make `dev` the default env of every shell
cat >> ~/.bashrc <<'RC'
if [ -d "$HOME/software/miniforge3/envs/dev" ]; then
    conda activate dev
fi
RC
```

Compiled packages (Corrfunc, classy, dsigma, pyfnntw, pymangle) are always
built on the machine itself, with `-march=native`, also from the locks.

## Use

```bash
./create.sh dev                 # solve + build + verify + write locks/
./create.sh dev-full
./create.sh jaxgpu              # GPU jax; see "The GPU env" below
./create.sh dev --force         # rebuild over an existing env
./create.sh dev --from-lock     # exact rebuild from locks/
./system.sh                     # once per machine: LaTeX (apt), Julia 1.11.9 (juliaup)
conda activate dev
python verify.py --env dev      # re-run the smoke tests any time
```

`create.sh` deactivates whatever env the calling shell has active before it
builds, and registers a Jupyter kernel named after the env (not for `jaxgpu`,
which has no ipykernel).

## How the repos use it

`dev` is *the* development env for every repo in `~/software`; no repo gets
its own env on this laptop.

- `~/.bashrc` runs `conda activate dev` (after the conda/mamba init blocks).
- Jupyter kernels `dev` and `dev-full` (`~/.local/share/jupyter/kernels/`).
- Local Makefiles and scripts default to
  `$(HOME)/software/miniforge3/envs/dev/bin/python`, still overridable
  (`make PY=...`).
- Each repo's README/CONTRIBUTING has a "Maintainer setup" note pointing
  here; its `environment.yml` stays for external users and is not used locally.
- Cluster scripts (oarsub/, slurm/, sciserver/) keep the cluster envs.
- GPU training runs in `jaxgpu`, e.g.
  `~/software/miniforge3/envs/jaxgpu/bin/python -m emu_pk.train ...`.

## Files

| file                     | role |
|--------------------------|------|
| `environment-dev.yml`    | conda-forge packages of `dev`: python, compilers, gsl/fftw/cfitsio/hdf5, jax, science stack, test/docs tools |
| `environment-extras.yml` | conda-forge packages added for `dev-full` (one solve with the above) |
| `pip-compiled.txt`       | compiled in the env: Corrfunc, classy, dsigma, pyfnntw, pymangle |
| `pip-pure.txt`           | not on conda-forge, pure Python (incl. git-only CEmulator, aemulusnu_hmf) |
| `pip-extras.txt`         | pure-Python extras for `dev-full` |
| `pip-extras-nodeps.txt`  | `dev-full` extras whose pins clash with the env, installed `--no-deps` (pyhalomodel) |
| `own-packages.txt`       | repos in `~/software` installed `pip install -e --no-deps`, in dependency order |
| `environment-jaxgpu.yml` | conda part of `jaxgpu`: python, pip, gcc/g++/make (for classy), nothing that imports numpy |
| `pip-jaxgpu.txt`         | pip part of `jaxgpu`: `jax[cuda13]`, optax, numpy, scipy, camb==1.6.6, GPU stress-test deps |
| `pip-jaxgpu-compiled.txt`| compiled in `jaxgpu`: classy 3.3.4 |
| `own-packages-jaxgpu.txt`| own packages of `jaxgpu` (emu_pk), editable `--no-deps` |
| `create.sh`              | builds an env; see the numbered steps inside |
| `verify.py`              | import + compiled smoke tests (Corrfunc, pyfnntw, classy, …; jax on the GPU for `jaxgpu`) |
| `lock_pip.py`            | writes the pip half of the locks |
| `locks/`                 | `<env>.conda.lock` (explicit, md5) + `<env>-pip*.txt`, rewritten by every build |

## The GPU env `jaxgpu`

The only env that runs jax on the NVIDIA GPU; `dev` and `dev-full` keep the
CPU jaxlib. It is built the way it was first made by hand (2026-10-01:
`conda create -n jaxgpu python=3.14`, then `pip install -U "jax[cuda13]"`):

- conda gives python, pip and the compilers (`environment-jaxgpu.yml`);
  everything else is pip (`pip-jaxgpu.txt`). `jax[cuda13]` brings jaxlib, the
  CUDA 13 plugin and the CUDA libraries as `nvidia-*` wheels, so the system
  needs only the NVIDIA driver, no CUDA toolkit.
- **camb is pinned to 1.6.6, with classy 3.3.4**: the emu_pk 2.1 training set
  was written on dahu with exactly these, and this env is where emu_pk 2.1 is
  validated against that truth. dev's camb 2.0.4 re-solves the training
  cosmologies up to 4.4e-4 off in ln P at k ~ 0.1 h/Mpc, too much for
  validation. Here, 6 re-solved training cosmologies match dahu's stored
  (float32) rows to |d ln P| <= 6.6e-7, the storage rounding (2026-10-05).
  Since 2026-10-06 dev and dev-full pin the same camb 1.6.6 (pip, as
  conda-forge stops at 1.6.5), so the technical paper's CAMB numbers and
  emu_pk's truth are one CAMB; their pyccl is pip too, because the
  conda-forge build depends on conda's camb.
- classy is compiled in the env (`pip-jaxgpu-compiled.txt`): the conda-forge
  gcc/g++ 15.3 (`create.sh` checks `gcc`/`g++` resolve into the env), an
  isolated build so setuptools/cython stay out of the env, and numpy pinned at
  build time to the env's numpy. CLASS's own flags: `-O3`, no OpenMP.
- **Never `mamba install` anything that imports numpy or jax into it** (the
  compilers are fine). A conda-forge package that depends on jax installs
  conda-forge jax and jaxlib over the pip ones: for optax the solver wanted
  jax 0.10.2 + a cuda130 jaxlib + 58 packages (2 GB), over pip jax 0.11.2.
- To add a package: put it in `pip-jaxgpu.txt` and `./create.sh jaxgpu --force`.
  To add it to the live env without a rebuild, pin what is there so nothing
  else moves, then rewrite the locks:

  ```bash
  PY=~/software/miniforge3/envs/jaxgpu/bin/python
  $PY -m pip install -c <($PY -m pip freeze --exclude-editable) <package>
  ~/software/miniforge3/bin/conda list -n jaxgpu --explicit --md5 > locks/jaxgpu.conda.lock
  PYTHONNOUSERSITE=1 $PY lock_pip.py pip-jaxgpu-compiled.txt locks/jaxgpu
  ```

- Own packages come from `own-packages-jaxgpu.txt` (only what needs nothing
  beyond `pip-jaxgpu.txt`), editable and built with isolation: the env has no
  setuptools.
- `python verify.py --env jaxgpu` fails unless jax sees a GPU device; it runs
  a matmul and 100 optax Adam steps on it, checks camb is 1.6.6 and runs one
  CAMB and one CLASS solve.

## Why compiled packages are done this way

Corrfunc previously failed to install on this laptop
(`cxg/python/all_galaxies/readme.sh`): the build mixed the system gcc and
headers with conda's gsl. The rule here is that **nothing compiled touches the
system toolchain or system libraries**:

- **Corrfunc** is built from the PyPI sdist, not taken from conda-forge: the
  conda-forge binary targets generic x86-64 and has no AVX/AVX2/AVX512
  kernels (it warns "CPU supports AVX2 but the compiler does not" on every
  call). `verify.py` fails if that warning appears.
- **Corrfunc, classy, dsigma, pyfnntw, pymangle** are compiled in the env. `create.sh`
  activates the env first — the `c-compiler`/`cxx-compiler`/`fortran-compiler`
  packages then set `CC`/`CXX`/`FC` to the env's gcc, and `rust` puts `rustc`
  and `cargo` on `PATH` — and runs `pip install --no-build-isolation`, so they
  compile against the env's numpy, gsl and OpenMP. `create.sh` also unsets
  `CFLAGS`, `LDFLAGS`, `CPATH`, `LD_LIBRARY_PATH`, … and aborts if
  `gsl-config` does not point into the env.
- The cluster recipe in `sum_stat/docs/guide/installation.rst` (build GSL by
  hand, sed `-march=native`) is not needed on the laptop: `-march=native`
  is correct for an env that only runs on this machine.

To add a package: prefer conda-forge (`environment-*.yml`); if it is only on
PyPI, put it in `pip-compiled.txt` when it has C/Cython/Rust/Fortran code,
else in `pip-pure.txt`; then `./create.sh dev --force`.

## Known caveats

- `classy` is capped `<3.4` (ggah_mod: the 3.4 sdist is broken).
- `pyhalomodel` requires Python `<3.13` and pins `numpy<2`; it is installed
  `--no-deps` and import-checked by `verify.py`.
- `jaxace` (via `quadax<0.3`) caps `numpy<2.5`; `AletheiaCosmo` pins
  `scikit-learn~=1.7.2`. Both pins live in `environment-dev.yml` so conda
  solves them; `create.sh` aborts if pip ever replaces a conda package.
- `ggah_mod_dev` has the same distribution name as `ggah_mod`; swap it in with
  `pip install -e ~/software/ggah_mod_dev --no-deps`.
- `dark_emulator` is not installed: 1.1.2 imports `scipy.misc.derivative`,
  removed in scipy 1.12 (used only by desi_erosita_agn's
  `validation_HOD_darkemu.py`).
- `dev-full` caps `scipy<1.18` for `eazy`.
- `mopc` (sum_stat SZ reference script) has no known source; not installed.
- sum_stat CI pins `numpy<2`, `jax<0.5`; this env uses current numpy/jax
  (rema and sys_mapping need `jax>=0.9`).
- Several Makefiles/READMEs still point at `~/mamba/envs/...` from the old
  laptop layout.
