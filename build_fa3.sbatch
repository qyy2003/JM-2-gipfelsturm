#!/bin/bash
#SBATCH --job-name=build-fa3
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460000
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -eo pipefail
mkdir -p logs

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FA3_PREFIX="/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3"
FA3_SRC="/iopsstor/scratch/cscs/$USER/gipfelsturm/flash-attention"
FA3_REF="${FA3_REF:-main}"

mkdir -p "$FA3_PREFIX" "$(dirname "$FA3_SRC")"

echo "START: $(date)"
echo "[fa3] prefix=$FA3_PREFIX  src=$FA3_SRC  ref=$FA3_REF"

# Clone or update source on the host (git inside the container is fine too,
# but doing it outside avoids re-cloning every run if you rerun this script).
if [ ! -d "$FA3_SRC/.git" ]; then
    git clone https://github.com/Dao-AILab/flash-attention.git "$FA3_SRC"
fi
git -C "$FA3_SRC" fetch --tags origin
git -C "$FA3_SRC" checkout "$FA3_REF"
git -C "$FA3_SRC" submodule update --init --recursive

srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     bash -c '
set -eo pipefail

echo "[fa3] in-container: $(hostname)"
python -c "import torch; print(\"[fa3] torch:\", torch.__version__, \"cuda:\", torch.version.cuda)"
nvcc --version | tail -n1

export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS='"$(( SLURM_CPUS_PER_TASK / 4 ))"'
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export NVCC_THREADS=2

cd '"$FA3_SRC"'/hopper

# Install into a host-mounted prefix so it survives container exit.
pip install --no-build-isolation --no-deps --target '"$FA3_PREFIX"' .

# Runtime deps (einops etc.) — also into the same target so PYTHONPATH alone is enough.
pip install --no-build-isolation --target '"$FA3_PREFIX"' einops

# Sanity: import + version. Newer FA3 wheels expose `flash_attn_3`; older
# ones used `flashattn_hopper`. Try both so this works across versions.
PYTHONPATH='"$FA3_PREFIX"':$PYTHONPATH python -c "
import importlib
for name in (\"flash_attn_3\", \"flashattn_hopper\"):
    try:
        m = importlib.import_module(name)
        fa = importlib.import_module(name + \".flash_attn_interface\")
        print(\"[fa3] import OK:\", name, getattr(m, \"__version__\", \"unknown\"))
        print(\"[fa3] module:\", fa.__file__)
        break
    except ModuleNotFoundError:
        continue
else:
    raise SystemExit(\"[fa3] neither flash_attn_3 nor flashattn_hopper importable\")
"
'

echo "[fa3] artifacts in: $FA3_PREFIX"
echo "[fa3] to use:  export PYTHONPATH=$FA3_PREFIX:\$PYTHONPATH"
echo "END: $(date)"
