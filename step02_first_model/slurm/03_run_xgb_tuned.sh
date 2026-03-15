#!/bin/bash
#SBATCH --job-name=antarctic_xgb_tuned
#SBATCH --account=uci157
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=xgb_tuned_pipeline_%j.out
#SBATCH --error=xgb_tuned_pipeline_%j.err

# === Environment ========================================================
module load singularitypro

# Lustre scratch for Spark local dirs and checkpoints
export LOCAL_SCRATCH=/expanse/lustre/projects/uci157/rrogers/temp
export SPARK_LOCAL_DIRS=/expanse/lustre/projects/uci157/rrogers/temp

# Make sure scratch dir exists
mkdir -p "$LOCAL_SCRATCH"

# Work from project directory
cd /expanse/lustre/projects/uci157/rrogers

echo "======================================================"
echo " Job ID        : $SLURM_JOB_ID"
echo " Node          : $(hostname)"
echo " CPUs          : $SLURM_CPUS_PER_TASK"
echo " Start         : $(date)"
echo " Python script : 03_xgb_tuned.py"
echo "======================================================"

# === Run ================================================================
singularity exec \
    --bind /expanse/lustre/projects/uci157/rrogers \
    ~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif \
    python 03_xgb_tuned.py

EXIT_CODE=$?

echo "======================================================"
echo " End           : $(date)"
echo " Exit code     : $EXIT_CODE"
echo "======================================================"

exit $EXIT_CODE