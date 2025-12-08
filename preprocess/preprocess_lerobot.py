"""
Code taken from:https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py
@article{black2410pi0,
  title={$pi$0: A vision-language-action flow model for general robot control. CoRR, abs/2410.24164, 2024. doi: 10.48550},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint ARXIV.2410.24164}
}
"""
import os
import cv2
import hydra
import shutil
import pickle
import natsort
import numpy as np
from pathlib import Path
from termcolor import colored
from omegaconf import DictConfig 

from utils import preprocess_video
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

@hydra.main(version_base="1.1", config_path="../conf", config_name="preprocess_lerobot")
def main(cfg: DictConfig) -> None:
    output_path = HF_LEROBOT_HOME / f"{cfg.task_name}_{cfg.num_demos}"
    # if output_path.exists():
    #     shutil.rmtree(output_path)
    if output_path.exists():
        shutil.rmtree(f"/home/daiyinlong/.cache/huggingface/lerobot/{cfg.task_name}_{cfg.num_demos}")

    # os.makedirs(output_path, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id = f"{cfg.task_name}_{cfg.num_demos}",
        robot_type = "panda",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            }
        },
        image_writer_threads = 10,
        image_writer_processes = 5
    )

    files = natsort.humansorted([f for f in os.listdir(cfg.data_path) if f.endswith('.pkl')])[:cfg.num_demos]
    print(files)
    for file in files: 
        print(colored("[Processing] ", "green") + f"Processing file: {file}")
        with open(os.path.join(cfg.data_path, file), "rb") as f:
            data = pickle.load(f)
        f.close() 

        data['img'] = np.clip(
            preprocess_video(
                np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                        for frame in np.array(data['img'])]),
                specs=cfg.video_specs
            ) / 255.0,
            0, 1
        )

        data['img_gripper'] = np.clip(
            preprocess_video(
                np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                        for frame in np.array(data['img_gripper'])]),
                specs=cfg.video_specs
            ) / 255.0,
            0, 1
        )
        # padd actions to length 32
        # pad_len = 32 - len(data['joint_vel'][0]) 
        # if pad_len > 0:
        #     data['joint_vel'] = [np.pad(arr, (0, pad_len), mode='constant') for arr in data['joint_vel']]

        for step in range(len(data["img"])):
            dataset.add_frame(
                {
                    "image": data["img"][step],
                    "wrist_image": data["img_gripper"][step],
                    "state": np.array(data["joint_state"][step], dtype = "float32"),
                    "actions": np.array(data["joint_vel"][step], dtype = "float32"),
                    "task": data["text_cond"] if "text_cond" in data.keys() else cfg.description
                }
            )
        dataset.save_episode()


if __name__ == "__main__":
    main()