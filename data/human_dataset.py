import os
import math
import torch
import einops
import pickle
import random
import numpy as np

from typing import Dict
from termcolor import colored
from collections import defaultdict
from torch.utils.data import Dataset
from scipy.interpolate import interp1d


class InterventionDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 window: int,
                 sample_num: int,
                 relative: bool = False) -> None:
        self.data_path = data_path
        self.window = window
        self.sample_num = sample_num + 1 if relative else sample_num
        self.relative = relative
        if self.window != -1:
            assert self.sample_num <= self.window, "Sample number must be less than or equal to the window size"

        self.files = [f for f in os.listdir(data_path) if f.endswith('_tracking.pkl')]
        for idx, file in enumerate(self.files):
            with open(os.path.join(data_path, file), "rb") as f:
                demo = pickle.load(f)
            if idx == 0:
                self.default_dict = defaultdict(list, demo)
                if 'text_cond' in demo:
                    self.default_dict['text_cond'] = [demo['text_cond']] * len(demo['observations'])
            else:
                for key, value in demo.items():
                    if key == 'text_cond':
                        self.default_dict[key].extend([value] * len(demo['observations']))
                        continue
                    self.default_dict[key].extend(value)
        self.preprocess_tracking() # preprocess tracking data
        print(colored("[Human Data] ", "green") + f"Loaded {len(self.default_dict['observations'])} interventions")
        if self.relative:
            print(colored("[Human Data] ", "yellow") + f"Using relative tracking data.")

    def preprocess_tracking(self) -> None:
        # Pad the tracking to the same length
        max_length = max(len(track) for track in self.default_dict['tracking'])
        tracking = []
        for i in range(len(self.default_dict['tracking'])):
            assert self.default_dict['tracking'][i].ndim == 3, f"Tracking data should be 3D, but got {self.default_dict['tracking'][i].ndim}D"
            if len(self.default_dict['tracking'][i]) < max_length:
                pad = np.zeros((max_length - len(self.default_dict['tracking'][i]), *self.default_dict['tracking'][i].shape[1:]), dtype=self.default_dict['tracking'][i].dtype)
                tracking.append(np.concatenate([self.default_dict['tracking'][i], pad], axis=0))
            else:
                tracking.append(self.default_dict['tracking'][i])
        tracking = np.stack(tracking)  # [B, T, N, 2]

        # Normalize the tracking,  tracking.shape (B, T, N, D)
        if self.relative:
            tracking = tracking[:, 1:, :, :] - tracking[:, :-1, :, :]  
            print(colored("[Human Data] ", "yellow") + f"Normalizing the tracking data to [-1, 1] range")
        else:
            print(colored("[Human Data] ", "green") + f"Normalizing the tracking data to [0, 1] range")
        tracking_reshaped = einops.rearrange(tracking, 'b t n d -> (b t n) d')
        self.max_flow = tracking_reshaped.max(axis=0)
        self.min_flow = tracking_reshaped.min(axis=0)
        print(colored("[Human Data] ", "cyan") + f"Max flow: {self.max_flow}, Min flow: {self.min_flow}")
        # tracking_reshaped = (tracking_reshaped - min_flow) / (max_flow - min_flow + 1e-8)
        # tracking = einops.rearrange(tracking_reshaped, '(b t n) d -> b t n d', b=B, t=T, n=N)

        # self.default_dict['tracking'] = list(tracking)

    def downsample_flow(self, 
                        flow: np.ndarray,
                        downsampled_horizon: int = 18) -> np.ndarray:
        t_flow = np.linspace(0, 1, flow.shape[0])
        t_downsampled = np.linspace(0, 1, downsampled_horizon)
        N = flow.shape[1]
        downsampled_flow = []

        for n in range(N):
            interp_x = interp1d(t_flow, flow[:, n, 0], kind='linear', fill_value='extrapolate')
            interp_y = interp1d(t_flow, flow[:, n, 1], kind='linear', fill_value='extrapolate')
            x_new = interp_x(t_downsampled)
            y_new = interp_y(t_downsampled)
            downsampled_flow.append(np.stack([x_new, y_new], axis=-1))  # shape: (T_obs, 2)
        downsampled_flow = np.stack(downsampled_flow, axis=1)  # shape: (T_obs, N, 2)

        return downsampled_flow

    def __len__(self) -> int:
        return len(self.default_dict['observations'])

    def __getitem__(self, idx: int) -> Dict:
        observations = self.default_dict['observations'][idx]
        tracking = self.default_dict['tracking'][idx]
        
        start_idx = random.randint(0, len(observations) - 1)
        sampled_observations = []
        sampled_tracking = []
        if self.window == -1:
            sampled_observations = observations
            sampled_tracking = tracking
        for i in range(self.window):
            if start_idx + i >= len(observations):
                # pad with the last observation
                sampled_observations.append(observations[-1])
                sampled_tracking.append(tracking[-1])
            else:
                sampled_observations.append(observations[start_idx + i])
                sampled_tracking.append(tracking[start_idx + i])
        
        indices = random.sample(range(1, len(sampled_observations) - 1), self.sample_num -2)
        indices = sorted([0] + indices + [len(sampled_observations) - 1])
        sampled_observations = torch.stack([torch.tensor(sampled_observations[i]) for i in indices]) / 255 # T H W C

        sampled_tracking = torch.stack([torch.tensor(sampled_tracking[i]) for i in indices]) # T N 2
        # sampled_tracking = torch.FloatTensor(self.downsample_flow(sampled_tracking, self.sample_num)) # T N 2

        # relative tracking
        if self.relative:
            sampled_tracking = sampled_tracking[1:, :, :] - sampled_tracking[:-1, :, :]
            sampled_tracking = torch.where(torch.abs(sampled_tracking) < 1, torch.zeros_like(sampled_tracking), sampled_tracking)
            # # normalize_to [-1,1]
            # min_max = np.concatenate([self.min_flow, self.max_flow], axis=0)
            # bound = abs(min_max).max(axis=0)
            # bound = 300 # NOTE: check if the bound is valid
            # sampled_tracking = (sampled_tracking / bound) * 500
        else:
            # sampled_tracking = (sampled_tracking - self.min_flow) / (self.max_flow - self.min_flow + 1e-8)  # Normalize to [0, 1]
            sampled_tracking = sampled_tracking

        sampled_observations = einops.rearrange(sampled_observations, 't h w c -> t c h w')
        sampled_tracking = einops.rearrange(sampled_tracking, 't (h w) d -> d t h w', h=int(math.sqrt(sampled_tracking.shape[1])), w=int(math.sqrt(sampled_tracking.shape[1])))
        item = {
            'observations': sampled_observations,
            'tracking': sampled_tracking
        }
        if 'text_cond' in self.default_dict:
            item['text_cond'] = self.default_dict['text_cond'][idx].squeeze()
        return item
    

class ImageDataset(Dataset): 
    def __init__(self, 
                 data_path: str) -> None:
        """ Different from Intervention Dataset, 
            this dataset gives one image at a time
        """
        self.data_path = data_path
        self.files = sorted([f for f in os.listdir(data_path) if f.endswith('_tracking.pkl')])

        self.max_demo_length = 0 # get max demo length
        for idx, file in enumerate(self.files):
            with open(os.path.join(data_path, file), "rb") as f:
                demo = pickle.load(f)
            total_length = sum(len(obs) for obs in demo['observations'])
            self.max_demo_length = max(self.max_demo_length, total_length)
        print(colored("[Human Data] ", "green") + f"Max demo length: {self.max_demo_length}")
        
        for idx, file in enumerate(self.files):
            with open(os.path.join(data_path, file), "rb") as f:
                demo = pickle.load(f)
            demo = self.get_certainty(demo)
            if idx == 0:
                self.default_dict = {}
                for key in demo.keys():
                    self.default_dict[key] = []
            for key, value in demo.items():
                if key == "text_cond": # text condition
                    for intervention in demo['observations']:
                        self.default_dict[key].extend([value] * len(intervention))
                    continue
                for intervention in value:
                    self.default_dict[key].extend(intervention)

        print(colored("[Human Data] ", "green") + f"Loaded {len(self.default_dict['observations'])} images")

    def get_certainty(self, demo: dict) -> dict:
        # Values should be decreasing
        # log decay
        # eps = 1e-4
        # T = np.linspace(eps, 1, self.max_demo_length + 1)
        # V = -np.log(T)

        # # exponential decay
        # V = np.exp(-np.linspace(0, 5, self.max_demo_length + 1)) 
        # V = V / V[0]

        # # Power decay
        T = np.linspace(0, 5, self.max_demo_length + 1)
        V = 10 * (1 - (T / 5) ** 2)
        
        demo['uncertainties'] = []
        idx = len(V)
        for intervention in demo['observations'][::-1]:
            uncertainties = V[idx - len(intervention):idx]
            assert len(uncertainties) == len(intervention), f"Uncertainties has length {len(uncertainties)}, but intervention has length {len(intervention)}"
            demo['uncertainties'] = [uncertainties] + demo['uncertainties']
            idx -= len(intervention)
        return demo
    
    def __len__(self) -> int:
        return len(self.default_dict['observations'])
    
    def __getitem__(self, idx: int) -> Dict:
        sampled_image = torch.tensor(self.default_dict['observations'][idx] / 255)
        sampled_uncertainty = torch.tensor(self.default_dict['uncertainties'][idx])

        sampled_image = einops.rearrange(sampled_image, 'h w c -> c h w')

        item = {
            'image': sampled_image,
            'uncertainty': sampled_uncertainty
        }
        if 'text_cond' in self.default_dict:
            item['text_cond'] = self.default_dict['text_cond'][idx].squeeze()
        return item


class HumanDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 window: int,
                 sample_num: int,
                 samplers_per_epoch: int,
                 data_type = 'intervention', # 'intervention' or 'image'
                 relative: bool = False) -> None:
        self.data_path = data_path
        self.samplers_per_epoch = samplers_per_epoch
        if data_type == 'image':
            self.data = ImageDataset(data_path)
        else:
            self.data = InterventionDataset(data_path, window, sample_num, relative=relative)

    def __len__(self) -> int:
        return self.samplers_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        sample_idx = random.randint(0, len(self.data) - 1)
        sample_item = self.data[sample_idx]
        return sample_item
    

def main():
    dataset = InterventionDataset(
        data_path="/projects/collab/Human2Robot/EaseScene/human_video/pick_berry",
        window = -1,
        sample_num = 17,
        relative=True)
    for i in range(len(dataset)):
        item = dataset[i]
        input ("*****************************")

if __name__ == "__main__":
    main()
