#!/bin/bash
#SBATCH --job-name=voronoi-workflow-large
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3:00:00
#SBATCH --output=voronoi-workflow-large-%j.out
#SBATCH --error=voronoi-workflow-large-%j.err

set -euo pipefail

module load miniforge
conda activate hpc_landlab
module load gcc/14.2.0
module load openmpi/5.0.6

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPI4PY_RC_IRECV_BUFSZ=1048576
export TMPDIR="$SLURM_SCRATCH"

cd /home/tiga7385/aspect_code/performance_evaluation

voronoi_timing_dir="voronoi_timing_results/job_${SLURM_JOB_ID}"

mpiexec -n "$SLURM_NTASKS" \
    python voronoi_lem_workflow.py \
    --grid-shape 2000,2000 \
    --seed 1 \
    --timing-output-dir "$voronoi_timing_dir"
