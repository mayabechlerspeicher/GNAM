#!/bin/bash
#SBATCH --job-name=cont_gnn
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

python main.py --model_name=gnam --data_name=cora --run_grid_search=0 --n_layers=3 --hidden_channels=64  --dropout=0.5 --early_stop=0