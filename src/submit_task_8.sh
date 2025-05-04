#!/bin/sh
#BSUB -q c02613
#BSUB -J task_8
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o batch_output/gpujob_%J.out
#BSUB -e batch_output/gpujob_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613
python task_8.py 50