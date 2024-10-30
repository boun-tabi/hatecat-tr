#!/bin/bash
#SBATCH --job-name=hatespan
#SBATCH --output=%j.log
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
##SBATCH --container-mounts /home/user/:/home/user/
#SBATCH --time=0-06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G


pip install -r requirements.txt 
python code.py
