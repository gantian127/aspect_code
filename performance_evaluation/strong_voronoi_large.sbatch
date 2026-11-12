#!/bin/bash
#SBATCH --job-name=strong-voronoi
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=strong-voronoi-%j.out
#SBATCH --error=strong-voronoi-%j.err

set -euo pipefail

module load miniforge
conda activate hpc_landlab
module load gcc/14.2.0
module load openmpi/5.0.6

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /home/tiga7385/aspect_code/performance_evaluation

python performance_strong_scaling.py \
    --workflow voronoi_lem_workflow.py \
    --grid-shape 2000,2000 \
    --cores 1 \
    --repetitions 1 \
    --log-x \
    --results-dir strong_scaling_results_voronoi_large_batch