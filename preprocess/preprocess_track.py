import os
import time
import hydra
import torch
import pickle 
import natsort
import numpy as np
from torch import Tensor
from typing import Tuple
from termcolor import colored

from omegaconf import DictConfig
from utils import preprocess_video, visualize_video
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

from transformers import AutoTokenizer, AutoModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true"


def generate_tracking(video: np.array, grid_size: int, visualize: bool, save_path: str, tracker, device) -> Tuple[np.array, np.array]:
    video = torch.tensor(video).float().to(device)
    try:
        assert len(video.shape) == 5, "Video should be in batch format (B, T, C, H, W)"
    except AssertionError as e:
        video = video.unsqueeze(0)
    try:
        assert video.shape[2] == 3, "Video should have 3 channels (RGB)"
    except AssertionError as e:
        video = video.permute(0, 1, 4, 2, 3)  

    pred_tracks, pred_visibility = tracker(video, grid_size=grid_size)
    # tracker(video_chunk=video, is_first_step=True, grid_size=grid_size)
    # for ind in range(0, video.shape[1] - tracker.step, tracker.step):
    #     pred_tracks, pred_visibility = tracker(
    #         video_chunk=video[:, ind : ind + tracker.step * 2]
    #     )  # B T N 2,  B T N 1
    

    if visualize:
        video = video.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()  # dimensions to (T, H, W, C)
        video = preprocess_video(video, specs={'cvtColor': 'BGR2RGB'})
        video = torch.tensor(video).unsqueeze(0).permute(0, 1, 4, 2, 3).float()  # back to (B, T, C, H, W)

        # vis = Visualizer(save_dir=f"{save_path}/tracking_videos", 
        #                  pad_value=120, 
        #                  linewidth=1, 
        #                  tracks_leave_trace=40, 
        #                  mode="optical_flow")
        # vis.visualize(
        #     video, pred_tracks.detach(), pred_visibility.detach(), filename=f"tracking_{time.time()}"
        # )

        vis = Visualizer(save_dir=f"{save_path}/paper_videos", 
                         pad_value=120, 
                         linewidth=1, 
                         tracks_leave_trace=-1, 
                         mode="rainbow")
        vis.visualize(
            video, pred_tracks.detach(), pred_visibility.detach(), filename=f"tracking_{time.time()}"
        )

        visualize_video(np.array(video), save_path=os.path.join(save_path, f"paper_videos/video_{time.time()}.mp4"))


    return pred_tracks.detach().cpu().numpy(), pred_visibility.detach().cpu().numpy()

@hydra.main(version_base="1.1", config_path="../conf", config_name="preprocess_track")
def main(cfg: DictConfig) -> None:

    # Load co-tracker model
    # tracker = CoTrackerOnlinePredictor(checkpoint=cfg.tracker_checkpoint).to(cfg.device)
    # print(colored("[Tracker] ", "green") + f"Loaded tracker model from {cfg.tracker_checkpoint}")
    tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(cfg.device)
    
    text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    files = natsort.humansorted([f for f in os.listdir(cfg.data_path) if f.endswith('.pkl')])
    for file in files:
        if "tracking" in file:
            print(colored("[Skipping] ", "yellow") + f"Skipping already processed file: {file}")
            continue
        print(colored("[Processing] ", "green") + f"Processing file: {file}")
        with open(os.path.join(cfg.data_path, file), "rb") as f:
            data = pickle.load(f)
        f.close()

        if 'observations' in data: # human dataset
            demo = data['observations'] if type(data['observations']) is list else [data['observations']]
            tracking = []
            augmented_obs = []
            for video in demo:

                # video = preprocess_video(video, specs=cfg.video_specs)
                video = preprocess_video(video, {"cvtColor": "BGR2RGB"})
                augmented_obs.append(video)
                video = torch.tensor(video).float().to(cfg.device)
                pred_tracks, _ = generate_tracking(video, cfg.grid_size, cfg.visualize, cfg.data_path, tracker, cfg.device)
                tracking.append(pred_tracks if pred_tracks.ndim == 3 else pred_tracks.squeeze(0))
            
            saved_data = {
                'observations': augmented_obs,
                'tracking': tracking
            }
            if 'text_cond' in data:
                
                inputs = tokenizer(data['text_cond'], return_tensors="pt", truncation=True, padding=True).to(next(text_encoder.parameters()).device)
                with torch.no_grad():
                    outputs = text_encoder(**inputs).last_hidden_state[:, 0, :]
                    saved_data['text_cond'] = outputs   
                
            # with open(os.path.join(cfg.data_path, file.replace('.pkl', '_tracking.pkl')), 'wb') as f:
            #     pickle.dump(saved_data, f)
            # print(colored("[Processing] ", "green") + f"Tracking data saved to {file.replace('.pkl', '_tracking.pkl')}")

        if 'joint_vel' in data: # robot dataset
            demo = data['img'] 
            inter_type = data['type'] if 'type' in data.keys() else None 
            print(colored("[Processing] ", "green") + f"Processing robot data of type: {inter_type}")
            # video = preprocess_video(np.array(demo), specs=cfg.video_specs)
            video = preprocess_video(np.array(demo), specs = {"cvtColor": "BGR2RGB"})
            gripper_video = preprocess_video(np.array(data['img_gripper']), specs=cfg.video_specs)

            pred_tracks, _ = generate_tracking(video, cfg.grid_size, cfg.visualize, cfg.data_path, tracker, cfg.device)
            pred_tracks = pred_tracks if pred_tracks.ndim == 3 else pred_tracks.squeeze(0)

            saved_data = {
                'type': inter_type,
                'img': video,
                'gripper_img': gripper_video,
                'joint_vel': data['joint_vel'],
                'joint_state': data['joint_state'],
                'tracking': pred_tracks
            }

            # with open(os.path.join(cfg.data_path, file.replace('.pkl', '_tracking.pkl')), 'wb') as f:
            #     pickle.dump(saved_data, f)
            # print(colored("[Processing] ", "green") + f"Tracking data saved to {file.replace('.pkl', '_tracking.pkl')}")

if __name__ == "__main__":
    main()
