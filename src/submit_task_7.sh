#!/bin/sh
#BSUB -q hpc
#BSUB -J task_7
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -W 40
#BSUB -o batch_output/gpujob_%J.out
#BSUB -e batch_output/gpujob_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613
python src/task_7.py