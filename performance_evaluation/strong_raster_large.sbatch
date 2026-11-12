#!/bin/bash
#SBATCH --job-name=strong-raster
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=strong-raster-%j.out
#SBATCH --error=strong-raster-%j.err

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
    --workflow raster_lem_example.py \
    --grid-shape 2000,2000 \
    --cores 1 2 4 8 16 32 64 \
    --repetitions 1 \
    --log-x \
    --results-dir strong_scaling_results_raster_large_batch \
    --workflow-args --mode d4 --seed 1