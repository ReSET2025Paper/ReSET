import os
import einops
import random
import pickle
import natsort
import numpy as np

from termcolor import colored
from collections import defaultdict
from torch.utils.data import Dataset

from utils import preprocess_video
from model.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

class RobotInterventionDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 sample_num: int,
                 video_specs: dict,
                 num_demos: int = None) -> None:
        """
        Data files should have the following structure:
        - pkl files containing:
            - img
            - img_gripper
            - joint_state
            - joint_vel
            - gripper_action
        - output structure:
            {
                'img': List of images (shape: [T, H, W, C]),
                'img_gripper': List of gripper images (shape: [T, H, W, C]),
                'joint_state': List of joint states (shape: [T, D]),
                'joint_vel': List of joint velocities (shape: [T, D]),
        """
        self.data_path = data_path
        self.sample_num = sample_num
        self.video_specs = video_specs
        self.files = natsort.humansorted([f for f in os.listdir(data_path) if f.endswith('.pkl')])
        if 'augmented' not in data_path and num_demos is not None:
            self.files = self.files[:num_demos]
        self.default_dict = defaultdict(list)
        for idx, file in enumerate(self.files):
            with open(os.path.join(data_path, file), "rb") as f:
                demo = pickle.load(f)
            for key, value in demo.items():
                if len(value) == 0 or value is None or not isinstance(value, list):
                    continue
                if key == "joint_state" and 'augmented' in data_path:
                    # append gripper state to joint state beacause we forgot to do so in the augmented data
                    value = [np.concatenate([state, [demo['gripper_state'][i]]]) for i, state in enumerate(value)]
                if key == "gripper_state" and 'augmented' in data_path:
                    key = "gripper_action"
                if key == "joint_vel" and 'augmented' in data_path:
                    # reduce the velocity except for the gripper
                    value = [np.concatenate([vel[:-1] * 0.05, [vel[-1]]]) for vel in value]
                if idx == 0:
                    self.default_dict[key] = [value]
                else:
                    self.default_dict[key].append(value)
        
        # check if using augmented data
        if 'augmented' in data_path:
            expert_data_path = data_path.replace('augmented', 'execution')
            expert_files = natsort.humansorted([f for f in os.listdir(expert_data_path) if f.endswith('.pkl')])
            if num_demos is not None:
                expert_files = expert_files[:num_demos]
            for idx, file in enumerate(expert_files):
                with open(os.path.join(expert_data_path, file), "rb") as f:
                    demo = pickle.load(f)
                for key, value in demo.items():
                    if len(value) == 0 or value is None or not isinstance(value, list):
                        continue
                    self.default_dict[key].append(value)

        print(colored("[Robot Data] ", "green") + f"Loaded {len(self.default_dict['img'])} robot demonstrations from {data_path}")

    def __len__(self) -> int:
        return len(self.default_dict["img"])

    def __getitem__(self, idx: int) -> dict:
        result_dict = {key:value[idx] for key, value in self.default_dict.items() if isinstance(value, list)}
        item = {key: []  for key in result_dict.keys()}

        start_idx = random.randint(0, len(result_dict['img']) - self.sample_num)
        for i in range(self.sample_num):
            for key, value in result_dict.items():
                if isinstance(value, list):
                    if start_idx + i < len(value):
                        item[key].append(value[start_idx + i])
                    else:
                        item[key].append(value[-1])
        for key in item.keys():
            item[key] = np.array(item[key])
        item['img'] = preprocess_video(item['img'], self.video_specs)
        item['img_gripper'] = preprocess_video(item['img_gripper'], self.video_specs)
        item['img'] = einops.rearrange(item['img'], 't h w c -> t c h w')
        item['img_gripper'] = einops.rearrange(item['img_gripper'], 't h w c -> t c h w')

        batch = {"obs": {
                    "agent_pos": item['joint_state'],
                    "image": item['img'],
                    "image_gripper": item['img_gripper'],
                },
        "action": item['joint_vel'],
        }

        return batch
    
class RobotDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 sample_num: int,
                 video_specs: dict,
                 samplers_per_epoch: int, 
                 num_demos: int = None) -> None:
        self.samplers_per_epoch = samplers_per_epoch
        self.data = RobotInterventionDataset(data_path, sample_num, video_specs, num_demos)

    def __len__(self) -> int:
        return self.samplers_per_epoch
    
    def __getitem__(self, idx: int) -> dict:
        sampled_idx = random.randint(0, len(self.data) - 1)
        sample_item = self.data[sampled_idx]
        return sample_item
    
    def get_normalizer(self, mode='limits'):
        all_actions = self.data.default_dict['joint_vel']
        all_actions = np.concatenate([np.array(demo) for demo in all_actions], axis=0)
        all_pose = self.data.default_dict['joint_state']
        all_pose = np.concatenate([np.array(demo) for demo in all_pose], axis=0)
        data = {
            'action': all_actions,
            'agent_pos': all_pose,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data, last_n_dims=1, mode=mode)

        scale = np.array([2 / 255], dtype=np.float32)
        offset = np.array([-1], dtype=np.float32)
        stat = {
            'min': np.array([0], dtype=np.float32),
            'max': np.array([255], dtype=np.float32),
            'mean': np.array([127.5], dtype=np.float32),
            'std': np.array([np.sqrt(1/12)], dtype=np.float32)
        }
        normalizer["image"] = SingleFieldLinearNormalizer.create_manual(
            scale=scale, 
            offset=offset, 
            input_stats_dict=stat
        )
        normalizer["image_gripper"] = SingleFieldLinearNormalizer.create_manual(
            scale=scale, 
            offset=offset, 
            input_stats_dict=stat
        )
        return normalizer
    
class PresetDataset(Dataset):
    def __init__(self,
                 data_path: str, # This is the path to the preset pickle file
                 video_specs: dict,
                 dynamics: bool = False,
                 flow_length: int = None,
                 custom_epoch_len: int = None) -> None:
        """
        - Output structure:
        {
            'img': single initial observation (shape: [H, W, C]),
            'correction_specs': 
                'type': 'pull'/'pick'/'rotate',
                'start_state': [x, y, z],
                'end_state': [x, y, z],}
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.video_specs = video_specs
        self.flow_length = flow_length
        self.action_types = {
            'pull': 0,
            'pick': 1,
            'rotate': 2
        }
        self.dynamics = dynamics
        self.custom_epoch_len = custom_epoch_len

    def preprocess_flow(self, flow: np.ndarray) -> np.ndarray:
        assert self.flow_length is not None, "Flow length must be specified for robot playdata."
        indices = random.sample(range(1, len(flow) - 1), self.flow_length - 1)
        indices = sorted([0] + indices + [len(flow) - 1])
        flow = flow[indices, :, :]  
        relative_tracking = flow[1:, :, :] - flow[:-1, :, :]
        relative_tracking = np.where(np.abs(relative_tracking) < 1, np.zeros_like(relative_tracking), relative_tracking)
        # bound = 300 # NOTE: check if the bound is valid
        # relative_tracking = (relative_tracking / bound) * 100
        # shape should be c t h w
        relative_tracking = einops.rearrange(relative_tracking, 't (h w) c -> c t h w', h=int(np.sqrt(relative_tracking.shape[1])))
        return relative_tracking


    def __len__(self) -> int:
        return len(self.data['correction_specs']) if self.custom_epoch_len is None else self.custom_epoch_len
    
    def __getitem__(self, idx: int) -> dict:
        if self.custom_epoch_len is not None:
            idx = idx % len(self.data['correction_specs'])
        correction_spec = self.data['correction_specs'][idx]
        img = preprocess_video(self.data['img'][idx], self.video_specs)
        img = einops.rearrange(img, 'h w c -> c h w')

        gt_type = self.action_types[correction_spec['type']]
        # one hot encode the type
        type_one_hot = np.zeros(len(self.action_types), dtype=np.float16)
        type_one_hot[gt_type] = 1.0


        if self.dynamics:
            obs = {}
            obs['img'] = img
            goal_img = preprocess_video(self.data['goal_img'][idx], self.video_specs)
            goal_img = einops.rearrange(goal_img, 'h w c -> c h w')
            obs['goal_img'] = goal_img
        elif 'flow' in self.data:
            obs = {}
            obs['flow'] = self.preprocess_flow(self.data['flow'][idx])
            obs['img'] = img
        else:
            obs = img

        batch = {
            'obs': obs,
            'action': {"type": type_one_hot,
                        "start_state": correction_spec['start_state'],
                        "end_state": correction_spec['end_state'],
                        "rotation": correction_spec['rotation']} 
        }
        return batch



def main():
    # data_path = "/projects/collab/Human2Robot/EaseScene/robot_video/pull_bowl/execution"
    # dataset = RobotInterventionDataset(data_path=data_path, sample_num=10)
    # print(dataset.default_dict.keys())
    # for i in range(len(dataset)):
    #     item = dataset[i]
    #     print(f"Item {i}: {item['img'].shape}, {item['img_gripper'].shape}, {item['joint_state'].shape}, {item['joint_vel'].shape}, {item['gripper_action'].shape}")

    data_path = "/projects/collab/Human2Robot/EaseScene/robot_video/pick_berry/playdata.pkl" 
    dataset = PresetDataset(data_path=data_path, video_specs=[224,224])
    for i in range(len(dataset)):
        item = dataset[i]
        # print(f"Item {i}: {item['img'].shape}, {item['correction_specs']}")
        input("************************************")

if __name__ == "__main__":
    main()