#!/bin/bash
#SBATCH --job-name=antarctic_eda_plots_only
#SBATCH --account=uci157
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=02_data_exploration_plots_only_%j.out
#SBATCH --error=02_data_exploration_plots_only_%j.err

# Environment
module load singularitypro

export PROJECT_DIR=/expanse/lustre/projects/uci157/rrogers
export LOCAL_SCRATCH=/expanse/lustre/projects/uci157/rrogers/temp
export SPARK_LOCAL_DIRS=$LOCAL_SCRATCH
export TMPDIR=$LOCAL_SCRATCH

mkdir -p "$LOCAL_SCRATCH"
cd "$PROJECT_DIR"

echo "======================================================"
echo " Job ID : $SLURM_JOB_ID"
echo " Node : $(hostname)"
echo " CPUs : $SLURM_CPUS_PER_TASK"
echo " Memory : 128G"
echo " Start : $(date)"
echo " Python script : 02_data_exploration_plots_only.py"
echo "======================================================"

# Run 
singularity exec \
    --bind /expanse/lustre/projects/uci157/rrogers \
    ~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif \
    python 02_data_exploration_plots_only.py

EXIT_CODE=$?

echo "======================================================"
echo " End           : $(date)"
echo " Exit code     : $EXIT_CODE"
echo "======================================================"

exit $EXIT_CODE
