#!/bin/bash
#SBATCH --job-name=weak-raster
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=weak-raster-%j.out
#SBATCH --error=weak-raster-%j.err

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

python performance_weak_scaling.py \
    --workflow raster_lem_example.py \
    --ranks 1 4 \
    --grid-shapes 300,300 600,600 \
    --repeats 3 \
    --log-x \
    --results-dir weak_scaling_results_raster_batch \
    --workflow-args --mode d4 --seed 1
