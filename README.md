# dev_env

Definition and creation of the conda environments used to develop every
project in `~/software` (not `~/software/previous/`).

| env        | what                                                                 |
|------------|----------------------------------------------------------------------|
| `dev`      | Python 3.12, CPU jax, all compiled deps, all own packages editable    |
| `dev-full` | `dev` + pyccl, NaMaster, galsim, yt/pyxsim, pytorch-cpu, notebook extras |
| `jaxgpu`   | separate, not managed here — GPU jax work                             |

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
./create.sh dev --force         # rebuild over an existing env
./create.sh dev --from-lock     # exact rebuild from locks/
./system.sh                     # once per machine: LaTeX (apt), Julia 1.11.9 (juliaup)
conda activate dev
python verify.py --env dev      # re-run the smoke tests any time
```

`create.sh` deactivates whatever env the calling shell has active before it
builds, and registers a Jupyter kernel named after the env.

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
| `create.sh`              | builds an env; see the numbered steps inside |
| `verify.py`              | import + compiled smoke tests (Corrfunc, pyfnntw, classy, …) |
| `lock_pip.py`            | writes the pip half of the locks |
| `locks/`                 | `<env>.conda.lock` (explicit, md5) + `<env>-pip*.txt`, rewritten by every build |

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
