#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --account=mdehghan_709
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:1
#SBATCH --error=errors/mfrc_final_%j.err
#SBATCH --out=out/mfrc_final_%j.out

module purge
module load gcc/11.3.0
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home1/sabdurah/.conda/envs/DT/lib/python3.7/site-packages/tensorrt
source /spack/conda/miniconda3/4.12.0/bin/activate
source activate mftc

python train_classifier.py "mfrc" "full" "final"
