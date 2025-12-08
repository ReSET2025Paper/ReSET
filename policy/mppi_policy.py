# This policy is implemented for the Dynamics-DP baseline.

"""
The baseline implemented is based on the paper:
@inproceedings{wu2025neural,
    title={Neural Dynamics Augmented Diffusion Policy},
    author={Wu, Ruihai and Chen, Haozhe and Zhang, Mingtong and Lu, Haoran and Li, Yitong and Li, Yunzhu},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
    year={2025}
}
"""
import os
import cv2
import hydra
import torch
import pickle
import einops
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple
from piq import psnr, ssim
from termcolor import colored
from omegaconf import OmegaConf

from . import Policy
from utils import FrankaResearch3, visualize_video

class MPPIPolicy(Policy):
    def __init__(self,
                 name: str,
                 dynamics_model_path: str = None,
                 play_data_path: str = None,
                 **kwags):
        super(MPPIPolicy, self).__init__(**kwags)
        self._initialize_models(dynamics_model_path)
        assert play_data_path is not None, "Correction path must be provided for MPPIPolicy."
        with open(play_data_path, 'rb') as f:
            play_data = pickle.load(f)
        print(play_data.keys())
        self.action_types = {
            'pull': 0,
            'pick': 1,
            'rotate': 2
        }
        self.action_map = {
            0: self._pull_seq,
            1: self._pick_seq,
            2: self._rotate_seq
        }
        self._get_action_and_goal(play_data)
        self._get_action_range()
        self.planner = MPPIPlanner(self.dynamics_model, 
                                   device = self.device,
                                   min_action=self.min_action,
                                   max_action=self.max_action)
        self.go_seq = None
        self.robot = None
        self.conn = None
        self.conn_gripper = None
        print(colored("[MPPI Policy] ", "green") + f"Loaded play data from {play_data_path}")

    def _get_action_and_goal(self, play_data) -> None:
        self.actions = []
        correction_specs = play_data["correction_specs"]
        for spec in correction_specs:
            gt_type = self.action_types[spec['type']]
            # one hot encode the type
            type_one_hot = np.zeros(len(self.action_types), dtype=np.float16)
            type_one_hot[gt_type] = 1.0

            self.actions.append(np.concatenate((
                spec['start_state'],
                spec['end_state'],
                np.expand_dims(spec['rotation'], axis=0),
                type_one_hot
            ), axis=0))
        self.actions = np.stack(self.actions)
        self.goal_imgs = play_data["goal_img"]
        self.imgs = play_data["img"]
        return

    def _get_action_range(self) -> None: # min action, max_action
        self.max_action = np.max(self.actions[:, :-3], axis=0)
        self.min_action = np.min(self.actions[:, :-3], axis=0)
        return

    def _initialize_models(self, dynamics_model_path: str = None) -> None:
        if dynamics_model_path is not None:
            if os.path.exists(f"{dynamics_model_path}/updated_config.yaml"):
                dynamics_cfg = OmegaConf.load(f"{dynamics_model_path}/updated_config.yaml")
            else:
                dynamics_cfg = OmegaConf.load(f"{dynamics_model_path}/.hydra/config.yaml")
            self.dynamics_model = hydra.utils.instantiate(dynamics_cfg.model).to(self.device)
            self.dynamics_model.load_state_dict(torch.load(f"{dynamics_model_path}/ckpt/model_best.pth", weights_only=True), strict=True)
            self.dynamics_model.eval()

    def init_robot(self, robot: FrankaResearch3 = None, conn = None, conn_gripper = None) -> None:
        self.robot = robot
        self.conn = conn
        self.conn_gripper = conn_gripper


    @torch.no_grad()
    def get_action(self, obs_dict: dict[str, np.array]) -> dict:
        if self.go_seq is None: 
            assert self.robot is not None and self.conn is not None and self.conn_gripper is not None, "Robot and connection must be initialized before getting action."
            home_joint = obs_dict["agent_pos"][-1]["q"]
            home_cart = self.robot.joint2pose(home_joint)[0]
            img = obs_dict['image'][-1] / 255
            self.planner.reset()
            self.planner.set_goal(self.goal_imgs, img)
            dynamics_output = self.planner.trajectory_optimization(obs_state=img, known_actions = self.actions, start_img = self.imgs)
            self.target_action = dynamics_output['start_state']
            self.go_seq = self.construct_go_seq(dynamics_output)
            self.go_seq.extend([{
                "goal": home_cart,
                "reached": False
            },
            {
                "goal": home_cart,
                "gripper": "open",
                "reached": False
            }])
        action = None
        gripper_action = 0
        gripper_action_count = 0
        rotation_target = None
        for idx, dict in enumerate(self.go_seq):
            if dict["reached"]:
                continue
            if "gripper" in dict:
                gripper_action = 0 if dict["gripper"] == "open" else 1
                action = np.zeros(7)
                if gripper_action_count >= 10:
                    self.go_seq[idx]["reached"] = True
                    gripper_action_count = 0
                else:
                    gripper_action_count += 1
                break
            if len(dict["goal"]) == 1:
                if rotation_target is None:
                    # NOTE: Unlike deploy.py, obs_dict["agent_pos"][-1] is everything reasState returns
                    rotation_target = obs_dict["agent_pos"][-1]["q"].copy()
                    rotation_target[-1] += dict["goal"]
                dist = np.linalg.norm(obs_dict["agent_pos"][-1]["q"] - rotation_target)
                action = np.clip(rotation_target - obs_dict["agent_pos"][-1]["q"], -0.1, 0.1)
                if dist < 0.02:
                    self.go_seq[idx]["reached"] = True
                break
            else:
                curr_goal = dict["goal"]
                joint_state = obs_dict["agent_pos"][-1]["q"]
                curr_cart_state = self.robot.joint2pose(joint_state)[0]
                # print("curr_cart_state: {}".format(curr_cart_state))
                # print("curr_goal: {}".format(curr_goal))

                dist = np.linalg.norm(curr_cart_state - curr_goal)   
                # xdot = np.clip(curr_goal - curr_cart_state, -0.05, 0.05)
                xdot = self.action_clip(curr_goal - curr_cart_state)
                xdot = np.concatenate([xdot, np.zeros(3)]) # add rotation to be 0
                action = self.robot.xdot2qdot(xdot, obs_dict["agent_pos"][-1])
                if dist < 0.02:
                    self.go_seq[idx]["reached"] = True
                    input("Temp goal reached!")
                break
                
        action = np.append(action, gripper_action)
        if all(d.get("reached", False) for d in self.go_seq):
            print(colored("[MPPIPolicy] ", "green") + f"Target reached! go_seq set to None!")
            self.go_seq = None
        return action if action is not None else self.go_seq
    
    def action_clip(self, action: np.array, limit: float = 0.05) -> np.array:
        """
        Clips the high fidelity joint action.
        """
        max_action_idx = np.argmax(np.abs(action))
        action_ratio = action / action[max_action_idx]
        clipped_action = np.clip(action, -limit, limit)
        clipped_action = action_ratio * clipped_action[max_action_idx]
        return clipped_action
        
    
    def construct_go_seq(self, dynamics_output: dict) -> list:
        """
        Output a list of dicts with struct:
        {
            "goal": np.array = cartesian goals,
            "gripper": str = "open" / "close"
            "reached": bool = False
        }
        """
        return self.action_map[dynamics_output['type']](dynamics_output)

    def _pull_seq(self, dynamics_output):
        go_seq = []
        go_seq.append({"goal": dynamics_output['start_state'] + np.array([0, 0, 0.05]),
                       "reached": False})
        go_seq.append({"goal": dynamics_output['start_state'] - np.array([0, 0, 0.02]),
                       "reached": False})
        end_state = dynamics_output['end_state']
        end_state[2] = dynamics_output['start_state'][2]
        go_seq.append({"goal": end_state, 
                       "reached": False})
        go_seq.append({"goal": end_state + np.array([0, 0, 0.05]),
                       "reached": False})
        return go_seq

    def _pick_seq(self, dynamics_output):
        go_seq = []
        go_seq.append({"goal": dynamics_output['start_state'] + np.array([0, 0, 0.05]),
                       "reached": False})
        go_seq.append({"goal": dynamics_output['start_state'] - np.array([0, 0, 0.02]),
                       "reached": False})
        go_seq.append({"goal": dynamics_output['start_state'] + np.array([0, 0, 0.02]),
                       "gripper": "close",
                       "reached": False})
        up_start_state = dynamics_output['start_state'] + np.array([0.0, 0.0, 0.1])
        up_end_state = dynamics_output['end_state'] + np.array([0.0, 0.0, 0.1])
        go_seq.append({"goal": up_start_state,
                       "reached": False})
        go_seq.append({"goal": up_end_state,
                       "reached": False})
        go_seq.append({"goal": dynamics_output['end_state'] - np.array([0, 0, 0.12]),
                       "reached": False})
        go_seq.append({"goal": dynamics_output['end_state'] - np.array([0, 0, 0.12]),
                       "gripper": "open",
                       "reached": False})
        go_seq.append({"goal": up_end_state,
                       "reached": False})
        return go_seq

    def _rotate_seq(self, dynamic_output):
        go_seq = []
        go_seq.append({"goal": dynamic_output['start_state'] + np.array([0, 0, 0.05]),
                       "reached": False})
        go_seq.append({"goal": dynamic_output['start_state'] - np.array([0, 0, 0.01]),
                       "reached": False})
        go_seq.append({"goal": dynamic_output['start_state'] - np.array([0, 0, 0.01]),
                       "gripper": "close",
                       "reached": False})
        go_seq.append({"goal": dynamic_output['start_state'] + np.array([0, 0, 0.05]),
                       "reached": False})
        go_seq.append({"goal": dynamic_output['rotation'],
                       "reached": False})
        go_seq.append({"goal": dynamic_output['rotation'],
                       "gripper": "open",
                       "reached": False})
        return go_seq



class MPPIPlanner():
    def __init__(self,
                 dynamics_model, 
                 device: torch.device,
                 num_samples: int = 60,
                 num_iters: int = 1,
                 max_action: np.ndarray = None,
                 min_action: np.ndarray = None
                 ) -> None:
        self.goal = None
        self.model = dynamics_model
        self.model.eval()
        self.device = device
        self.max_action = max_action
        self.min_action = min_action
        self.noise_level = 0.05
        self.num_samples = num_samples
        self.num_iters = num_iters
        self.reset()
        
    def reset(self) -> None:
        self.goal = None
        pass

    def get_img_similarity(self, input_img1: torch.Tensor, input_img2: torch.Tensor) -> float:
        img1 = input_img1.clone().detach().to(self.device)
        img2 = input_img2.clone().detach().to(self.device)
        assert img1.shape == img2.shape, "Image shapes do not match."
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        if img1.shape[-1] == 3:
            img1 = img1.permute(0, 3, 1, 2)
            img2 = img2.permute(0, 3, 1, 2)
        img_psnr = psnr(img1, img2, reduction = "none")
        img_ssim = ssim(img1, img2, reduction = "none")
        if img_psnr.shape[0] == 1:
            psnr_norm = img_psnr
            ssim_norm = img_ssim
        else:
            psnr_norm = (img_psnr - img_psnr.min()) / (img_psnr.max() - img_psnr.min() + 1e-18)
            ssim_norm = (img_ssim - img_ssim.min()) / (img_ssim.max() - img_ssim.min() + 1e-18)
        return psnr_norm, ssim_norm
    
    def set_goal(self, goals: np.ndarray, obs: np.ndarray) -> None:
        highest_similarity = 0.0
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        # for image in goals:
        #     image = torch.tensor(image, dtype=torch.float32, device=self.device) / 255
        #     psnr, ssim = self.get_img_similarity(obs, image)
        #     similaity = psnr + ssim
        #     if similaity > highest_similarity:
        #         highest_similarity = similaity
        #         self.goal = image
        
        obs = torch.stack([obs] * len(goals), dim=0)
        psnr, ssim = self.get_img_similarity(obs, torch.Tensor(goals) / 255)
        similarity = psnr
        self.goal = torch.Tensor(goals[np.argmax(similarity.cpu().numpy())]) / 255
        # vis_goal = np.array(self.goal.clone().detach())
        # cv2.imshow("goal", vis_goal)
        # cv2.waitKey(0)


    def evaluate_traj(self, states: torch.Tensor, show_values: bool = False) -> np.ndarray: # return an array of awards
        goals = torch.stack([self.goal] * len(states), dim=0)
        img_psnr, img_ssim = self.get_img_similarity(states.squeeze(1), goals) # squeeze the time dimension
        # if show_values:
        #     print(f"Image PSNR: {img_psnr}, SSIM: {img_ssim}")
        # reward_seqs.append(img_psnr.detach().cpu() + img_ssim.detach().cpu())
        reward_seqs = img_psnr.unsqueeze(0).detach().cpu()
        reward_seqs = np.array(reward_seqs)
        return reward_seqs
    
    def trajectory_optimization(self, obs_state: np.ndarray, known_actions: np.array = None, start_img: np.array = None) -> dict: # action sequence
        sample = np.random.uniform(low=self.min_action, 
                                    high=self.max_action
                )
        # indices = np.random.choice(3, self.num_samples, p = np.array([1/3, 1/3, 1/3]))
        action_init = np.concatenate((sample, np.array([1, 1, 1])), axis=0)
        obs_states = torch.stack([torch.tensor(obs_state, dtype=torch.float32, device=self.device)] * self.num_samples)

        # cv2.imshow("current img", obs_state)
        # cv2.waitKey(0)
        # action_seq = torch.Tensor(known_actions)
        # for action, img in zip(action_seq, start_img):
        #     cv2.imshow("dataset img", img / 255)
        #     cv2.waitKey(0)
        #     print(img)
        #     known_action_out = self.model(
        #         # {'img': torch.tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0)},
        #         {'img': torch.tensor(obs_state, dtype=torch.float32, device=self.device).unsqueeze(0) * 255},
        #         action.unsqueeze(0)
        #     )
        #     known_action_out = known_action_out.squeeze().detach().cpu()
        #     known_action_out = np.asarray(known_action_out)
        #     print(known_action_out.shape)
        #     cv2.imshow(f"Dynamics ouput", known_action_out)
        #     cv2.waitKey(0)


        with torch.no_grad():
            prbar = tqdm(total=self.num_iters, desc="Optimizing Actions...")
            action = action_init
            output_list = []
            reward_list = []
            for i in range(self.num_iters):
                action_seq = torch.Tensor(self.sample_action_sequence(action))
                model_out = self.model({'img': obs_states * 255}, action_seq) 
                reward_seqs = self.evaluate_traj(model_out)
                # avg_reward = np.mean(reward_seqs)
                action = self.optimize_action(action_seq, reward_seqs)

                curr_action_result = self.model({'img': torch.Tensor(obs_state).unsqueeze(0) * 255}, torch.Tensor(action).unsqueeze(0))
                vis_output = curr_action_result.squeeze(0).squeeze(1).detach().cpu()
                vis_output = np.asarray(vis_output)
                cv2.imshow(f"Dynamics ouput", vis_output[0])
                cv2.waitKey(1)
                curr_reward = self.evaluate_traj(curr_action_result, show_values = True)
                output_list.append(vis_output)
                reward_list.append(curr_reward)
                prbar.set_postfix(curr_reward=curr_reward)
                prbar.update(1)
            prbar.close()

        # save images and rewards
        save_path = "/home/collab/Yinlong/EaseScene/temp_video.mp4"
        visualize_video(np.array(output_list), save_path)
        # save rewards as a plot
        from matplotlib import pyplot as plt
        plt.plot(reward_list)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Reward vs Iteration")
        plt.savefig("/home/collab/Yinlong/EaseScene/temp_reward_plot.png")
        plt.close()

        # NOTE: This is just for debugging
        # action = known_actions[3]
        
        start_state = action[:3]
        end_state = action[3:6]
        rotation = action[6]
        type = np.argmax(action[7:])
        # input("*************************************")

        return {"action": action,
                "start_state": start_state,
                "end_state": end_state,
                "rotation": rotation,
                "type": type,
                "model_outputs": model_out,
                "eval_outputs": reward_seqs}
    

    def sample_action_sequence(self, action: np.ndarray) -> np.ndarray:
        """
        Sample a seq of N actions based on the given action.
        """
        # if type(action) is not torch.Tensor:
        #     action = torch.tensor(action, dtype=torch.float32, device=self.device) 
        # assert type(action) is torch.Tensor, "Action must be a torch tensor."

        action_seq = np.stack([action] *self.num_samples)

        # sample different action types based on the action type distribution
        probs = action[len(self.min_action):] / action[len(self.min_action):].sum()
        print(probs)
        indices = np.random.choice(len(probs), size=self.num_samples, p=probs)
        type_seq = np.eye(len(probs))[indices]

        # sample noise for the continuous action dimensions
        noise_sampled = torch.normal(0.0, self.noise_level, (self.num_samples, len(self.min_action))).to(self.device)
        noise_sampled = noise_sampled.cpu().numpy()

        action_seq[:, :len(self.min_action)] = action_seq[:, :len(self.min_action)] + noise_sampled
        action_seq = torch.Tensor(action_seq)
        action_seq[:, -len(type_seq[0]):] = torch.tensor(type_seq, dtype=torch.float32, device=self.device)
        return action_seq

    def optimize_action(self, action_seq: torch.Tensor, reward_seqs: np.ndarray) -> np.ndarray:
        """
        Get one optimized action.
        (N, action_dim) collapsed to (action_dim)
        """
        reward_seqs = einops.rearrange(reward_seqs, 'n c -> c n')
        action = torch.sum(action_seq * F.softmax(torch.Tensor(reward_seqs), dim=0), dim=0)
        action = action.cpu().numpy()
        clipped_action = self.clip_action(action[:len(self.min_action)])
        action = np.concatenate((clipped_action, action[len(self.min_action):]), axis = 0)
        return action
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip the action to be within the action bounds.
        The action input does not include last 3 dimensions for action type. (start pos, end pos, rotation)
        """
        assert action.shape == self.min_action.shape, "Action shape does not match the expected shape."
        clipped_action = np.clip(action, self.min_action, self.max_action)
        return clipped_action

