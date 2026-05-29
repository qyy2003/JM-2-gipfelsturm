#!/bin/bash
# Export nsys reports to sqlite for the overlap analysis.
#   sbatch export.sbatch <rep_dir> <out_dir> [node0 node4 ...]
# Example (this run):
#   sbatch export.sbatch \
#     /iopsstor/scratch/cscs/course_00293/gipfelsturm/gipfelsturm/profile-8b-8n/nsys/2406105 \
#     /iopsstor/scratch/cscs/course_00293/JM-2-gipfelsturm/visual/nsys_work
# Must run via sbatch (not a bare login-node srun): the pyxis/PMIx appdir is
# only set up correctly inside a batch allocation.
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --job-name=nsys_export
#SBATCH --output=nsys_export-%j.log
#SBATCH --no-requeue
HERE=/iopsstor/scratch/cscs/course_00293/JM-2-gipfelsturm/visual/nsys_work
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
     --cpus-per-task "$SLURM_CPUS_PER_TASK" --wait 60 \
     bash "$HERE/run_export.sh" "$@"
echo "srun exit code: $?"
