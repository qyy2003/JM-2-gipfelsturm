#!/bin/bash
#SBATCH --job-name=build-fa4
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200000
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -eo pipefail
mkdir -p logs

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FA4_PREFIX="/iopsstor/scratch/cscs/$USER/gipfelsturm/fa4"
# Upstream recommends flash-attn-4[cu13] on CUDA 13, but current beta releases
# (4.0.0b3..b13 on PyPI) don't define that extra, so use the base package.
# Pinning to a specific beta avoids surprise upgrades.
FA4_PKG="${FA4_PKG:-flash-attn-4==4.0.0b13}"

mkdir -p "$FA4_PREFIX"

echo "START: $(date)"
echo "[fa4] prefix=$FA4_PREFIX  pkg=$FA4_PKG"

# FA4 is CuTeDSL-based (Python + JIT-to-PTX at runtime), so unlike FA3 we
# don't need git clone / submodules / nvcc — pip install is enough.
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     bash -c '
set -eo pipefail

echo "[fa4] in-container: $(hostname)"
python -c "import torch; print(\"[fa4] torch:\", torch.__version__, \"cuda:\", torch.version.cuda)"

# Install into a host-mounted prefix so it survives container exit. Use the
# package name (with extras) directly — no source build needed.
pip install --no-build-isolation --target '"$FA4_PREFIX"' "'"$FA4_PKG"'"

# Sanity: import + version. FA4 exposes the kernels under flash_attn.cute.
PYTHONPATH='"$FA4_PREFIX"':$PYTHONPATH python -c "
import flash_attn
from flash_attn.cute import flash_attn_func  # noqa: F401
print(\"[fa4] import OK: flash_attn\", getattr(flash_attn, \"__version__\", \"unknown\"))
print(\"[fa4] flash_attn_func from flash_attn.cute is importable\")
"
'

echo "[fa4] artifacts in: $FA4_PREFIX"
echo "[fa4] to use:  export PYTHONPATH=$FA4_PREFIX:\$PYTHONPATH"
echo "END: $(date)"
