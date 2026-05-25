<h1 align="center" style="font-size: 2.0em; font-weight: bold; margin-bottom: 0; border: none; border-bottom: none;">Prepare Before You Act: 

Learning From Humans to Rearrange Initial States</h1>

##### <p align="center"> [Yinlong Dai](https://yinlongdai.github.io/), [Andre Keyser](https://www.linkedin.com/in/andre-keyser-560090380/), [Dylan P. Losey](https://dylanlosey.com/)</p>
##### <p align="center"> Collaborative Robotics Lab (Collab), Virginia Tech </p>

#####
<div align="center">
    <a href="https://reset2025paper.github.io"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Website&color=blue"></a> &ensp;
    <a href="https://arxiv.org/abs/2509.18043"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red"></a> &ensp; 
</div>


> **ReSET** combines human videos and teleoperation data to predict simplifying actions to handle out-of-distribution environments by first rearranging initial scenes back into the training distribution. 

## Setup Instructions

### 1. Installation & Environment Synchronization
Clone the repository and synchronize the dependencies:
```bash
git clone https://github.com/ReSET2025Paper/ReSET.git ReSET
cd ReSET

uv sync
```

### 2. Download Pretrained CoTracker3 Checkpoints  [[repo](https://github.com/facebookresearch/co-tracker.git)]
Create the `checkpoints` directory at the project root and download the CoTracker3 checkpoints:
```bash
mkdir -p checkpoints
cd checkpoints

# Download online model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth

# Download offline model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth

cd ..
```

## Pipeline Execution

### 1. Data Collection
Data collection scripts are stored under `record` folder. This implementation uses a Franka robot and webcams to record videos, see `FrankaResearch3` and `camera` class in `utils.py`. To record human videos or robot play demonstrations:
* **Collect Human Correction Videos**:
  ```bash
  uv run record/record_human.py task=[task_name] multi_task=false
  ```
* **Collect Robot Correction Play Data**:
  ```bash
  uv run record/record_robot.py task=[task_name] type=correction
  ```
  *(Saves raw correction demonstration pickled `.pkl` datasets into the configured `./robot_video/[task_name]/correction` folder).*
* **Collect Robot Teleoperation Demos (Non-Correction / Execution)**:
  ```bash
  uv run record/record_robot.py task=[task_name] type=execution
  ```
  *(Saves raw execution demonstration pickled `.pkl` datasets into the configured `./robot_video/[task_name]/execution` folder).*

### 2. Track Preprocessing (Flow Generation)
Preprocess scripts for generating flow with Cotracker and Lerobot dataset are located under `preprocess`.

Compute dense optical flow trajectories using the offline CoTracker model:
```bash
uv run preprocess/preprocess_track.py task=[task_name] data_path=./human_video/[task_name] 

# or

uv run preprocess/preprocess_track.py task=[task_name] data_path=./robot_video/[task_name]/correction 
```
*(Processes the raw human/robot demonstration files in `[task_name]/correction/` and outputs corresponding `*_tracking.pkl` files).*

### 3. Play Data Preprocessing
* **Preprocess and Align (preset mode)**:
  ```bash
  uv run preprocess_play_data.py task=[task_name] playdata_type=preset
  ```
* **Preprocess Robot Playdata**:
  ```bash
  uv run preprocess_play_data.py task=[task_name] playdata_type=robot
  ```
* **Combine/Aggregate All Tasks**:
  ```bash
  uv run preprocess_play_data.py task=all playdata_type=robot
  uv run preprocess_play_data.py task=all playdata_type=preset
  ```


## Policy Training & Deployment

All main training and prediction scripts are organized under the `train/` directory. For dataset structure, please refer to comments at the beginning of each script. For multi-GPU (`torchrun`) and SLURM job execution templates, please refer directly to [train_example.sh](train_example.sh).

### Primary Training Scripts
1. **Flow Prediction (`train/pred_track_vit.py`)**
   - Trains the flow prediction decoder (`FlowDecoder`) to estimate keypoint-wise 2D trajectory flows from initial observations.
2. **Uncertainty Estimation (`train/pred_uncertainty.py`)**
   - Trains the uncertainty network (`LinearEstimator`) using to predict intervention boundaries relative to human teleoperation thresholds.
3. **Policy Training (`train/train_policy.py`)**
   - Trains both correction and base policies. e.g., `FlowPolicy`, `DiffusionPolicy` or `PresetPolicy` (for ReSET Naive baseline).

### Evaluate Policy Deployment
Run evaluation and hardware deployment using the root deployment script:
```bash
uv run python deploy.py --config-name=deploy task=[task_name]
```

## Citation

If you find this work userful, please consider citing:

```bibtex
@article{dai2025prepare,
  title={Prepare before you act: Learning from humans to rearrange initial states},
  author={Dai, Yinlong and Keyser, Andre and Losey, Dylan P},
  journal={arXiv preprint arXiv:2509.18043},
  year={2025}
}
```

