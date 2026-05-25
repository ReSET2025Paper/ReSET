#!/bin/bash
#SBATCH -J ReSET_Train
#SBATCH --account=your_slurm_account
#SBATCH --partition=your_slurm_partition
#SBATCH --nodes=1 --cpus-per-task=4
#SBATCH --time=0-15:00:00
#SBATCH --gres=gpu:4 --ntasks-per-node=4
#SBATCH --mem=200G
#SBATCH --output=outputs/job_outputs/%x_%j.out

# Collect environment variables and logging parameters
EXTRA_ARGS="$@"
LOG_FILE="outputs/job_logs.txt"
USER_ID=$(id -un)

mkdir -p outputs/job_outputs

echo " " >> "$LOG_FILE"
echo "User: $USER_ID" >> "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID | Submission Time: $(date)" >> "$LOG_FILE"
echo "Arguments: $EXTRA_ARGS" >> "$LOG_FILE"

# Make sure we are in the repository root directory
cd "$(dirname "$0")"
# Set debug environment variables
export HYDRA_FULL_ERROR=1

# Resolve job identifier
JOB_IDENTIFIER=${SLURM_JOB_ID:-123}

# --- 1. Train Flow Prediction (pred_track_vit.py) ---
torchrun \
    --nproc_per_node=4 \
    --master_port=$((12000 + RANDOM % 1000)) \
    train/pred_track_vit.py \
    task=your_task_name \
    dataset.data_path=/path/to/your/human_video/your_task_name \
    user_name="$USER_ID" \
    job_id="$JOB_IDENTIFIER" \
    distributed=True \
    +dataset.relative=True

# --- 2. Train Robot Policy (train_policy.py) ---
# Option A: Train corrective FlowPolicy (default)
# torchrun \
#     --nproc_per_node=4 \
#     --master_port=$((12000 + RANDOM % 1000)) \
#     train/train_policy.py \
#     dataset.data_path=/path/to/your/robot_video/playdata.pkl \
#     user_name="$USER_ID" \
#     job_id="$JOB_IDENTIFIER" \
#     distributed=True \
#     batch_size=32

# Option B: Train DiffusionPolicy 
# torchrun \
#     --nproc_per_node=4 \
#     --master_port=$((12000 + RANDOM % 1000)) \
#     train/train_policy.py \
#     policy=Diffusion \
#     use_ema=True \
#     dataset=robot \
#     dataset.data_path=/path/to/your/robot_video/your_task_name/execution \
#     user_name="$USER_ID" \
#     job_id="$JOB_IDENTIFIER" \
#     distributed=True \
#     batch_size=32

# Option C: Train PresetPolicy (For ReSET Naive)
# python train/train_policy.py \
#     policy=PresetPolicy \
#     dataset=preset \
#     dataset.data_path=/path/to/your/robot_video/action_preset.pkl \
#     user_name="$USER_ID" \
#     job_id="$JOB_IDENTIFIER" \
#     epochs=200

# --- 3. Train Observation Dynamics Predictor (pred_dynamics.py) ---
# torchrun \
#     --nproc_per_node=4 \
#     --master_port=$((12000 + RANDOM % 1000)) \
#     train/pred_dynamics.py \
#     dataset.data_path=/path/to/your/robot_video/playdata.pkl \
#     user_name="$USER_ID" \
#     job_id="$JOB_IDENTIFIER" \
#     distributed=True \
#     epochs=600

# --- 4. Train Uncertainty/Intervention Estimator (pred_uncertainty.py) ---
# python train/pred_uncertainty.py \
#     task=your_task_name \
#     dataset.data_path=/path/to/your/human_video/your_task_name \
#     user_name="$USER_ID" \
#     job_id="$JOB_IDENTIFIER" \
#     epochs=200
