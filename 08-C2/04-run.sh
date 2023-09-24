#!/bin/bash
#SBATCH --job-name ST00-07-04
#SBATCH -o ST00-07-04-%j.out
#SBATCH --nodes=1
#SBATCH --time=5-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=icelake
#SBATCH --no-requeue
#SBATCH --mem=200G

source ~/.bashrc

module purge

XPWD=`pwd`

echo "===================================================="
echo "        Job ID is:         $SLURM_JOBID"
echo "        Job name is:       $SLURM_JOB_NAME"
echo "        Hostname is:       "`hostname`
echo "        This dir is:       $XPWD"
echo "        CPUs per Task is:  $SLURM_CPUS_PER_TASK"
echo "        Tasks per Node is: $SLURM_TASKS_PER_NODE"
echo "        Tasks per Node is: $SLURM_MEM_PER_CPU"
echo "        Tasks per Node is: $SLURM_MEM_PER_GPU"
echo "        Tasks per Node is: $SLURM_MEM_PER_NODE"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export LD_PRELOAD=

IX=${SLURM_ARRAY_TASK_ID}

which orterun
which python3

python3 -u 04-pdmrg.py > 04-pdmrg.out
