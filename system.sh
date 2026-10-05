#!/usr/bin/env bash
# Tools that do not belong in a conda env.  Run once per machine.
#
#   LaTeX  -- for the *_paper / *_technical_paper repos and erc-cog-prop.
#             conda-forge's texlive is too incomplete for those Makefiles.
#   Julia  -- for ggah_mod_benchmark (SymBoltz, pinned to Julia 1.11.9).
#             juliaup handles several Julia versions side by side, which the
#             single conda-forge `julia` package cannot.
set -euo pipefail

echo "==> LaTeX (needs sudo)"
sudo apt-get update
sudo apt-get install -y texlive-full latexmk curl   # curl: for juliaup below

echo "==> Julia via juliaup"
if ! command -v juliaup >/dev/null; then
  # Official installer; adds ~/.juliaup/bin to PATH in the shell rc files.
  curl -fsSL https://install.julialang.org | sh -s -- --yes
  export PATH="$HOME/.juliaup/bin:$PATH"
fi
juliaup add 1.11.9
echo "Julia versions:"; juliaup status
