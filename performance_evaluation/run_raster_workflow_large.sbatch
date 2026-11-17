#!/bin/bash
#SBATCH --job-name=raster-workflow-large
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=raster-workflow-large-%j.out
#SBATCH --error=raster-workflow-large-%j.err

set -euo pipefail

module load miniforge
conda activate hpc_landlab
module load gcc/14.2.0
module load openmpi/5.0.6

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$SLURM_SCRATCH"

cd /home/tiga7385/aspect_code/performance_evaluation

raster_timing_dir="raster_timing_results/job_${SLURM_JOB_ID}"

mpiexec -n "$SLURM_NTASKS" \
    python raster_lem_example.py \
    --grid-shape 2000,2000 \
    --mode d4 \
    --seed 1 \
    --timing-output-dir "$raster_timing_dir"
