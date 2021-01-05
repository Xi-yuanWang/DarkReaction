#! /usr/bin/bash
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1800011848@pku.edu.cn
#SBATCH -J test
#SBATCH -o out
#SBATCH -e error
#SBATCH --time=48:00:00
#SBATCH --qos=normal
conda init bash
conda activate torch_geo
python SVC.py