#!/bin/bash
#SBATCH -J FlowGen
#SBATCH --account=collab
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1 --cpus-per-task=4
#SBATCH --time=0-15:00:00
#SBATCH --gres=gpu:4 --ntasks-per-node=4
#SBATCH --mem=200G
#SBATCH --output=/projects/collab/Human2Robot/EaseScene/outputs/job_outputs/%x_%j.out

EXTRA_ARGS="$@"
LOG_FILE="log_file.txt"
USER_ID=$(id -un)

echo " " >> "$LOG_FILE"
echo "$USER_ID" >> "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID: Submission Time: $(date)" >> "$LOG_FILE"
echo "$EXTRA_ARGS" >> "$LOG_FILE" # This saves your specs to log_file.txt


cd /projects/collab/Human2Robot/LatentAction
pwd

module reset
# module load site/tinkercliffs/easybuild/setup
module load Miniconda3
source activate /projects/collab/env/RSCENE
cd /projects/collab/Human2Robot/EaseScene\

mkdir -p /projects/collab/Human2Robot/LatentAction/training_outputs/job_outputs
export HYDRA_FULL_ERROR=1
# python3 train.py user_name=$USER_ID job_id=$SLURM_JOB_ID model=bootstrap_lam

torchrun \
    --nproc_per_node=4 \
    --master_port=$((12000 + RANDOM % 1000)) \
    pred_track_vit.py user_name=$USER_ID job_id=$SLURM_JOB_ID distributed=True +dataset.relative=True 
    # train_policy.py user_name=$USER_ID job_id=$SLURM_JOB_ID distributed=True batch_size=32
    # train_flow_vae.py user_name=$USER_ID job_id=$SLURM_JOB_ID distributed=True +dataset.relative=True



    
    
    