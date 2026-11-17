#!/bin/bash
#SBATCH --job-name=raster-original-large
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=raster-original-large-%j.out
#SBATCH --error=raster-original-large-%j.err

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

workflow_script="/home/tiga7385/aspect_code/performance_evaluation/run_lem_example_ori.py"
workflow_output_dir="/home/tiga7385/aspect_code/performance_evaluation/original_workflow_results/job_${SLURM_JOB_ID}"

mkdir -p "$workflow_output_dir"
cd "$workflow_output_dir"

mpiexec -n "$SLURM_NTASKS" \
    python "$workflow_script" \
    --grid-shape 2000,2000 \
    --mode d4 \
    --seed 1
