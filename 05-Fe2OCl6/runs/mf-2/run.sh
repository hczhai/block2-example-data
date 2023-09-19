#!/bin/bash

#SBATCH --job-name fe2o.2.hife
#SBATCH -o LOG.%j
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --partition=any
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --no-requeue
#SBATCH --mem=120G

source ~/.bashrc

module purge
module load gcc/9.2.0

export PYSCF_TMPDIR=/central/scratch/hczhai/hife/fe2ocl6-def2-SVP-uks.1/runs/mf-2

which python3
python3 --version
python3 -c "import pyscf; print(pyscf.__version__)"
python3 -c "import pyscf; print(pyscf.__file__)"
python3 -c "import pyscf; print(pyscf.lib.param.TMPDIR)"
python3 -c "import block2; print(block2.__file__)"
python3 -c "import pyblock2; print(pyblock2.__file__)"

echo SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
echo SLURM_JOBID=$SLURM_JOBID
echo SLURM_JOB_NAME=$SLURM_JOB_NAME
echo HOST_NAME = $(hostname)
echo PWD = $(pwd)
echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export XRUN=orterun
export PYSCF_MPIPREFIX="$XRUN --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$SLURM_CPUS_PER_TASK"


if [ "${SLURM_CPUS_PER_TASK}" != "" ]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

if [ "0" = "1" ]; then
    SCPT=dmrg
    if [ "0" = "1" ]; then
        SCPT=dmrg-rev
    fi
else
    SCPT=hife
fi

TJ=$(echo ${SCPT}.out.* | tr ' ' '\n' | grep '\*$' -v | wc -l)
export TJ=$(expr ${TJ} + 1)
echo ${SCPT}.out.${TJ} >> OUTFILE
echo $SLURM_JOBID >> JOBIDS

which $XRUN

if [ "$?" = "1" ] || [ "${SLURM_TASKS_PER_NODE}" = "" ] || [ "0" = "1" ]; then
    if [ "0" = "1" ]; then
        [ -f ./FCIDUMP ] && rm ./FCIDUMP
        ln -s /central/scratch/hczhai/hife/fe2ocl6-def2-SVP-uks.1/runs/mf-2/FCIDUMP ./FCIDUMP
        cp /central/scratch/hczhai/hife/fe2ocl6-def2-SVP-uks.1/runs/mf-2/${SCPT}.conf ${SCPT}.conf.${TJ}
        python3 -u $(which block2main) ${SCPT}.conf.${TJ} > ${SCPT}.out.${TJ}
    else
        python3 -u ${SCPT}.py 0 > ${SCPT}.out.${TJ}
    fi
else
    if [ "0" = "1" ]; then
        [ -f ./FCIDUMP ] && rm ./FCIDUMP
        ln -s /central/scratch/hczhai/hife/fe2ocl6-def2-SVP-uks.1/runs/mf-2/FCIDUMP ./FCIDUMP
        cp /central/scratch/hczhai/hife/fe2ocl6-def2-SVP-uks.1/runs/mf-2/${SCPT}.conf ${SCPT}.conf.${TJ}
        if [ "$XRUN" = "srun" ]; then
            srun python3 -u $(which block2main) ${SCPT}.conf.${TJ} > ${SCPT}.out.${TJ}
        else
            $XRUN --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS \
                python3 -u $(which block2main) ${SCPT}.conf.${TJ} > ${SCPT}.out.${TJ}
        fi
    else
        if [ "$XRUN" = "srun" ]; then
            srun python3 -u ${SCPT}.py 0 > ${SCPT}.out.${TJ}
        else
            $XRUN --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS \
                python3 -u ${SCPT}.py 0 > ${SCPT}.out.${TJ}
        fi
    fi
fi

if [ "$?" = "0" ]; then
    echo "SUCCESSFUL TERMINATION"
else
    echo "ERROR TERMINATION"
fi

if [ "0" = "1" ]; then
    cp /central/scratch/hczhai/hife/fe2ocl6-def2-SVP-uks.1/runs/mf-2/node0/1pdm.npy ${SCPT}.1pdm.${TJ}.npy
fi