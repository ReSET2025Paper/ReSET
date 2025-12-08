import os
import einops
import random
import pickle
import h5py
import tqdm
import natsort
import numpy as np

from termcolor import colored
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image

from utils import preprocess_video
from model.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

class HumanoidInterventionDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 sample_num: int,
                 video_specs: dict) -> None:
        """
        Data files should have the following structure:
        - hdf5 files containing:
            - colors: {color_0, color_1}
            - states: {left arm joint states, right arm joint states, left ee joint state, right ee joint states}
            - actions: {left arm joint velocities, right arm joint velocities, left arm joint torques, right arm joint torques, left ee joint velocities, right ee joint velocities}
        - output structure:
            {
                'robot_img': List of robot head camera images (shape: [T, H, W, C]),
                'static_img': List of static camera images (shape: [T, H, W, C]),
                'joint_state': List of joint states (shape: [T, D]),
                'joint_vel': List of joint velocities (shape: [T, D]),
        """
        self.data_path = data_path
        self.sample_num = sample_num
        self.video_specs = video_specs
        self.files = []
        self.color0 = []
        self.color1 = []
        color_0_list = []
        color_1_list = []
        
        for folder in os.listdir(data_path):
            print(folder)
            self.files.append(([data_path + '/' +  folder + '/' + file for file in os.listdir(data_path + '/' + folder) if file.endswith('.hdf5')][0]))
            colors = data_path + '/' + folder + '/colors/'
            color_0_list.append(natsort.humansorted([colors + file for file in os.listdir(colors) if file.endswith('0.jpg')]))
            color_1_list.append(natsort.humansorted([colors + file for file in os.listdir(colors) if file.endswith('1.jpg')]))

            color_0_list = natsort.humansorted(color_0_list)
            color_1_list = natsort.humansorted(color_1_list)

        with tqdm.tqdm(total = len(color_0_list)) as pbar:
            for episode in color_0_list:
                img_list = []
                for img in episode:
                    img_array = Image.open(img)
                    img_array = preprocess_video(np.array(img_array), self.video_specs)
                    img_list.append(img_array)
                self.color0.append(img_list)
                pbar.update(1)
        with tqdm.tqdm(total = len(color_1_list)) as pbar:
            for episode in color_1_list:
                img_list = []
                for img in episode:
                    img_array = Image.open(img)
                    img_array = preprocess_video(np.array(img_array), self.video_specs)
                    img_list.append(img_array)
                self.color1.append(img_list)
                pbar.update(1)
            
        self.files = natsort.humansorted(self.files)
        self.default_dict = defaultdict(list)
        for idx, file in enumerate(self.files):
            with h5py.File(file) as f:
                states = f['states']
                actions = f['actions']

                left_arm_states = np.array(states['left arm joint states'][()])
                right_arm_states = np.array(states['right arm joint states'][()])
                left_ee_states = np.array(states['left ee joint states'][()])
                right_ee_states = np.array(states['right ee joint states'][()])

                left_arm_velocities = np.array(actions['left arm joint velocities'][()])
                right_arm_velocities = np.array(actions['right arm joint velocities'][()])
                left_ee_velocities = np.array(actions['left ee joint velocities'][()])
                right_ee_velocities = np.array(actions['right ee joint velocities'][()])
        
                f.close()

            states_list = []
            velocities_list = []

            for id in range(len(left_arm_states)):
                states_list.append(np.concatenate((left_arm_states[id], right_arm_states[id], left_ee_states[id], right_ee_states[id])))
                velocities_list.append(np.concatenate((left_arm_velocities[id], right_arm_velocities[id], left_arm_states[id] + left_ee_velocities[id], right_arm_velocities[id])))

            demo = {
                    'robot_img': self.color0[idx],
                    'static_img': self.color1[idx],
                    'states': states_list,
                    'actions': velocities_list
            }
            for key, value in demo.items():
                if len(value) == 0 or value is None or not isinstance(value, list):
                    continue
                if idx == 0:
                    self.default_dict[key] = [value]
                else:
                    self.default_dict[key].append(value)

        print(colored("[Robot Data] ", "green") + f"Loaded {len(self.default_dict['robot_img'])} robot demonstrations from {data_path}")

    def __len__(self) -> int:
        return len(self.default_dict["robot_img"])

    def __getitem__(self, idx: int) -> dict:
        result_dict = {key:value[idx] for key, value in self.default_dict.items() if isinstance(value, list)}
        item = {key: []  for key in result_dict.keys()}

        start_idx = random.randint(0, len(result_dict['robot_img']) - self.sample_num)
        for i in range(self.sample_num):
            for key, value in result_dict.items():
                if isinstance(value, list):
                    if start_idx + i < len(value):
                        item[key].append(np.stack((value[start_idx + i]), axis=0))
                    else:
                        item[key].append(np.stack(value[-1]), axis=0)

        # item['robot_img'] = np.stack(item['robot_img'], axis=0)
        # item['static_img'] = np.stack(item['static_img'], axis=0)

        for key in item.keys():
            item[key] = np.array(item[key])
        
        # item['robot_img'] = preprocess_video(item['robot_img'], self.video_specs)
        # item['static_img'] = preprocess_video(item['static_img'], self.video_specs)
        item['robot_img'] = einops.rearrange(item['robot_img'], 't h w c -> t c h w')
        item['static_img'] = einops.rearrange(item['static_img'], 't h w c -> t c h w')

        batch = {"obs": {
                    "joint_state": item['states'],
                    "robot_img": item['robot_img'],
                    "static_img": item['static_img'],
                },
        "joint_vel": item['actions'],
        }

        return batch
    
# def main():
#     data_path = '/projects/collab/Human2Robot/EaseScene/robot_video/move_banana'
#     dataset = HumanoidInterventionDataset(data_path=data_path, sample_num=4, video_specs={'resize':[224, 224]})
#     print(f"length: {len(dataset)}")
#     for i in range(len(dataset)):
#         item = dataset[i]
#         print(f"Item {i}: {item['obs']['robot_img'].shape}, {item['obs']['static_img'].shape}, {item['obs']['joint_state'].shape}, {item['joint_vel'].shape}")

# if __name__ == "__main__":
#     main()