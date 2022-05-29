#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1



# only use the following on partition with GPUs

#SBATCH --job-name="ncspp"
#SBATCH --output=/atlas/u/aamdekar/slurm_jobs/score/ncsn-%j.out

# only use the following if you want email notification
#SBATCH --mail-user="aamdekar@cs.stanford.edu"
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
# NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
# echo NPROCS=$NPROCS
source /sailhome/aamdekar/.bashrc
#source /sailhome/cundy/.zshrc

conda activate jax_ncsn
cd /atlas/u/aamdekar/score_matching_refined/ || exit
python main.py use_wandb=True num_epochs=100
