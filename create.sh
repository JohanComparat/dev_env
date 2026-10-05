#!/usr/bin/env bash
# Build (or rebuild) a development environment.
#
#   ./create.sh dev                 # core env, solved from environment-dev.yml
#   ./create.sh dev-full            # dev + environment-extras.yml, one solve
#   ./create.sh jaxgpu              # GPU jax: environment-jaxgpu.yml + pip-jaxgpu*.txt
#   ./create.sh dev --from-lock     # exact rebuild from locks/
#   ./create.sh dev --force         # remove an existing env of that name first
#
# Every successful build rewrites locks/<env>.* and runs verify.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFTWARE="$(dirname "$HERE")"
CONDA_ROOT="${CONDA_ROOT:-$SOFTWARE/miniforge3}"
MAMBA="$CONDA_ROOT/bin/mamba"

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

ENV=""; FROM_LOCK=0; FORCE=0
for a in "$@"; do
  case "$a" in
    dev|dev-full|jaxgpu) ENV="$a" ;;
    --from-lock)  FROM_LOCK=1 ;;
    --force)      FORCE=1 ;;
    *)            usage ;;
  esac
done
[[ -n "$ENV" ]] || usage
LOCK="$HERE/locks/$ENV"
COMPILED_TXT="$HERE/pip-compiled.txt"
[[ "$ENV" == jaxgpu ]] && COMPILED_TXT="$HERE/pip-jaxgpu-compiled.txt"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# --- 1. clean shell: nothing from the system may leak into the builds --------
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH LIBRARY_PATH \
      LD_LIBRARY_PATH PKG_CONFIG_PATH PYTHONPATH PIP_REQUIRE_VIRTUALENV
export PYTHONNOUSERSITE=1          # ignore ~/.local/lib/python3.12
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
# ~/.bashrc activates `dev`: leave every env (this subshell only), so the
# active env's PATH and compiler variables cannot leak into the build.
set +u; while (( ${CONDA_SHLVL:-0} > 0 )); do conda deactivate; done; set -u

if (( FROM_LOCK )) && ! grep -qs "^@EXPLICIT" "$LOCK.conda.lock"; then
  echo "$LOCK.conda.lock is missing or not an @EXPLICIT lock; nothing removed." >&2; exit 1
fi
if "$MAMBA" env list | awk '{print $1}' | grep -qx "$ENV"; then
  if (( FORCE )); then
    say "Removing existing env $ENV"; "$MAMBA" env remove -y -n "$ENV"
  else
    echo "Env '$ENV' exists; pass --force to rebuild it." >&2; exit 1
  fi
fi

# --- 2. conda part -----------------------------------------------------------
if (( FROM_LOCK )); then
  say "Creating $ENV from $LOCK.conda.lock"
  "$CONDA_ROOT/bin/conda" create -y -n "$ENV" --file "$LOCK.conda.lock"
else
  SPEC="$HERE/environment-dev.yml"
  if [[ "$ENV" == jaxgpu ]]; then
    SPEC="$HERE/environment-jaxgpu.yml"
  elif [[ "$ENV" == dev-full ]]; then
    # One solve over dev + extras: append the extras' dependency lines.
    SPEC="$(mktemp --suffix=.yml)"; trap 'rm -f "$SPEC"' EXIT
    { cat "$HERE/environment-dev.yml"; echo
      sed -n '/^dependencies:/,$p' "$HERE/environment-extras.yml" | tail -n +2
    } > "$SPEC"
  fi
  say "Solving and creating $ENV"
  "$MAMBA" env create -y -n "$ENV" -f "$SPEC"
fi

# --- 3. activate: the compiler packages set CC/CXX/FC to the env's gcc -------
set +u; conda activate "$ENV"; set -u
if [[ "$ENV" == jaxgpu ]]; then     # classy's Makefile/setup.py call gcc, g++ by name
  say "Toolchain: gcc=$(command -v gcc)  g++=$(command -v g++)"
  [[ "$(command -v gcc)" == "$CONDA_PREFIX/bin/gcc" && "$(command -v g++)" == "$CONDA_PREFIX/bin/g++" ]] \
    || { echo "gcc/g++ do not come from $CONDA_PREFIX" >&2; exit 1; }
else
  say "Toolchain: CC=$CC  FC=$FC  rustc=$(command -v rustc)"
  [[ "$(gsl-config --prefix)" == "$CONDA_PREFIX" ]] \
    || { echo "gsl-config does not point into $CONDA_PREFIX" >&2; exit 1; }
fi
PIP=(python -m pip --disable-pip-version-check)

# --- 4. pip part ---------------------------------------------------------------
if [[ "$ENV" == jaxgpu ]]; then
  # jax, its CUDA 13 wheels and the rest: pip only (see environment-jaxgpu.yml).
  if (( FROM_LOCK )); then
    say "pip packages from lock"
    "${PIP[@]}" install --no-deps -r "$LOCK-pip.txt"
    COMPILED_REQ="$LOCK-pip-compiled.txt"
  else
    say "jax[cuda13] and the other pip packages"
    "${PIP[@]}" install -r "$HERE/pip-jaxgpu.txt"
    COMPILED_REQ="$COMPILED_TXT"
  fi
  # Isolated build (setuptools/cython stay out of the env) against the env's
  # own numpy, with the env's gcc (checked in step 3).
  say "Compiled pip packages (conda gcc, isolated build on the env's numpy)"
  BUILD_PINS="$(mktemp)"; trap 'rm -f "$BUILD_PINS"' EXIT
  python -c 'import numpy; print(f"numpy=={numpy.__version__}")' > "$BUILD_PINS"
  "${PIP[@]}" install --no-deps --build-constraint "$BUILD_PINS" -r "$COMPILED_REQ"
elif (( FROM_LOCK )); then
  say "Compiled pip packages from lock"
  "${PIP[@]}" install --no-build-isolation --no-deps -r "$LOCK-pip-compiled.txt"
  say "Pure pip packages from lock"
  "${PIP[@]}" install --no-deps -r "$LOCK-pip.txt"
else
  say "Compiled pip packages (conda gcc/rustc, no build isolation)"
  "${PIP[@]}" install --no-build-isolation -r "$HERE/pip-compiled.txt"
  say "Pure pip packages"
  "${PIP[@]}" install -r "$HERE/pip-pure.txt"
  if [[ "$ENV" == dev-full ]]; then
    "${PIP[@]}" install -r "$HERE/pip-extras.txt"
    "${PIP[@]}" install --no-deps -r "$HERE/pip-extras-nodeps.txt"
  fi
fi

# pip must only add packages: one that replaced a conda package means a pin
# belongs in environment-*.yml instead.
say "Checking pip did not replace any conda package"
python - "$CONDA_PREFIX" <<'EOF'
import json, re, sys
from importlib.metadata import distributions
from pathlib import Path
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
conda = {norm(json.loads(p.read_text())["name"]) for p in Path(sys.argv[1], "conda-meta").glob("*.json")}
bad = [f"{d.metadata['Name']} {d.version}" for d in distributions()
       if (d.read_text("INSTALLER") or "").strip() == "pip" and norm(d.metadata["Name"]) in conda]
if bad:
    sys.exit("pip replaced conda packages: " + ", ".join(sorted(bad))
             + "\n-> pin them in environment-*.yml so conda solves them.")
print("  ok")
EOF

# --- 5. own packages, editable -----------------------------------------------
# jaxgpu gets its own short list, built with isolation: it has no setuptools,
# and those packages are pure Python.
OWN_LIST="$HERE/own-packages.txt"; NO_ISOLATION=(--no-build-isolation)
if [[ "$ENV" == jaxgpu ]]; then
  OWN_LIST="$HERE/own-packages-jaxgpu.txt"; NO_ISOLATION=()
fi
say "Own packages (editable, --no-deps)"
FAILED=()
while read -r repo; do
  if [[ ! -d "$SOFTWARE/$repo" ]]; then
    echo "  $repo: not cloned in $SOFTWARE, skipped" >&2
  elif "${PIP[@]}" install "${NO_ISOLATION[@]}" --no-deps -q -e "$SOFTWARE/$repo" \
         > "/tmp/dev_env-$repo.log" 2>&1; then
    echo "  $repo"
  else
    echo "  $repo: FAILED (log: /tmp/dev_env-$repo.log)" >&2; FAILED+=("$repo")
  fi
done < <(grep -vE '^\s*(#|$)' "$OWN_LIST")

# --- 6. checks -----------------------------------------------------------------
say "pip check"
"${PIP[@]}" check || echo "(pip check reported conflicts -- see above)" >&2
say "verify.py"
VERIFY_OK=1
python "$HERE/verify.py" --env "$ENV" || VERIFY_OK=0

# --- 7. locks --------------------------------------------------------------------
if (( ! FROM_LOCK )); then
  say "Writing locks/$ENV.*"
  mkdir -p "$HERE/locks"
  # conda, not mamba: mamba 2 prints a table, not an @EXPLICIT url list.
  "$CONDA_ROOT/bin/conda" list -n "$ENV" --explicit --md5 > "$LOCK.conda.lock"
  grep -q "^@EXPLICIT" "$LOCK.conda.lock" || { echo "bad conda lock" >&2; exit 1; }
  python "$HERE/lock_pip.py" "$COMPILED_TXT" "$LOCK"
fi
if [[ "$ENV" != jaxgpu ]]; then     # jaxgpu has no ipykernel
  say "Jupyter kernel '$ENV'"
  python -m ipykernel install --user --name "$ENV" --display-name "Python ($ENV)"
fi

say "Done: conda activate $ENV"
if (( ${#FAILED[@]} )); then
  echo "Own packages that did not install: ${FAILED[*]}" >&2
fi
(( VERIFY_OK )) || echo "verify.py reported failures (see above)." >&2
(( VERIFY_OK && ! ${#FAILED[@]} )) || exit 2
