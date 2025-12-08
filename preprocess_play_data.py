import os
import cv2
import time
import glob
import tqdm
import hydra
import einops
import pickle
import natsort
import numpy as np
from termcolor import colored
from omegaconf import DictConfig
from scipy.interpolate import interp1d
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

from utils import visualize_pred_tracking, save_to_gif

def downsample_flow(flow: np.ndarray,
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


def get_closest_action_flow_pair(flow: np.array,  
                                 human_init_obs: np.array,
                                 play_data: np.array, 
                                 horizon: int = 18) -> int:
    closest_dist = float('inf')
    closest_flow = None

    # NOTE: Here assume both flow and play_flow are absolute flow as output from cotracker model
    assert flow.ndim == 3 and flow.shape[-1] == 2, 'Flow shape should be (T H*W C)'
    human_flow = downsample_flow(flow, downsampled_horizon=horizon) 
    # human_init_obs = rgb2gray(human_init_obs.copy())

    for idx, data in play_data.items():
        robot_flow = data['robot_tracking']  # shape: (T, H*W, 2)
        robot_img = data['robot_img'][0]  # shape: (H, W, C)
        robot_flow = downsample_flow(robot_flow, downsampled_horizon=horizon)
        flow_dist = np.linalg.norm(human_flow - robot_flow) 

        # robot_img = rgb2gray(robot_img.copy())
        # img_dist = ssim(human_init_obs, robot_img, data_range=255)
        img_dist = np.linalg.norm(human_init_obs - robot_img) / 15
        print(f"Flow dist: {flow_dist}, Img dist: {img_dist}")
        dist = flow_dist + img_dist
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = idx
            closest_flow = robot_flow
    print(f"closest_idx:{closest_idx}")

    return closest_idx, closest_flow, human_flow

def get_obj_indices(flow: np.ndarray, init_obs: np.ndarray = None) -> np.ndarray:
    # # Get the set of points with the most distplacement
    # # NOTE: Assuming the flow is absolute
    # init_points = flow[0, :, :]  
    # final_points = flow[-1, :, :] 
    # displacement = np.linalg.norm(final_points - init_points, axis=-1)  
    # # get point indices with displacement higher than a threshold
    # obj_indices = np.where(abs(displacement) > 10)[0]     # drop_bowl: 10; pick_berry: 50

    # Get the set of points with most average velocity
    init_points = flow[:-1, :, :]  # shape: (T-1, N, 2)
    final_points = flow[1:, :, :]  # shape: (T-1, N, 2)
    # total_displacement = np.sum(np.linalg.norm(final_points - init_points, axis=-1), axis=0)  # shape: (N,)
    max_displacement = np.max(np.linalg.norm(final_points - init_points, axis=-1), axis=0)  # shape: (N,)
    # get point indices with total displacement higher than a threshold
    # obj_indices = np.where(total_displacement > 70)[0] # bowl: 80
    obj_indices = np.where(max_displacement > 1.2)[0] # multi: 1 screw, block: 1.2

    # Get the highest and lowest x, y coordinates of the points
    assert len(obj_indices) > 0, "No object detected in the flow"

    init_points = flow[0, :, :]
    if init_obs is not None:
        # visualize the bounding box
        obj_coords = init_points[obj_indices]
        x_min, y_min = np.min(obj_coords, axis=0)
        x_max, y_max = np.max(obj_coords, axis=0)
        bbox = np.array([x_min - 10, y_min - 10, x_max + 10, y_max + 10], dtype=np.int32)
        flow = downsample_flow(flow)
        flow_vis = visualize_pred_tracking(init_obs,
                                            einops.rearrange(flow, 't n c -> n t c'),
                                            normalized = False)
        os.makedirs(f"obj_bbox", exist_ok=True)
        save_to_gif(flow_vis,
                    save_path=f"obj_bbox/bbox_{time.time()}.gif",
                    # bbox=bbox)
                    bbox = None)
        print(colored(f'[PlayData] ', 'yellow') + f"Visualizing obj interaction!!")
    return obj_indices

def get_interaction_specs_for_obj(robot_demo: dict, robot = None) -> int:
    interaction_specs  = {"type": None,
                          "start_state": None,
                          "end_state": None}
    # Get interaction type
    if "type" in robot_demo.keys():
        assert robot_demo["type"] in ["pull", "pick", "rotate"], f"Invalid interaction type: {robot_demo['type']}"
        interaction_specs['type'] = robot_demo['type']
    
    # Get obj indices
    flow = robot_demo['robot_tracking']
    obj_indices = get_obj_indices(flow, init_obs=robot_demo['robot_img'][0])
    # obj_indices = get_obj_indices(flow)
    
    # Get interaction starting point and ending point
    start_timestep = 0
    avg_displacement = np.mean(np.linalg.norm(
            robot_demo['robot_tracking'][1:, obj_indices] - robot_demo['robot_tracking'][:-1, obj_indices], axis=-1), axis=1)

    timesteps = np.where(avg_displacement > 0.5)[0] #  block: 0.5, screw: 0.6 multi: 0.5, bowl: 0.5
    assert len(timesteps) > 0, "No interaction detected in the flow"
    start_timestep, end_timestep = timesteps[0], timesteps[-1]

    assert robot is not None, "Must have a robot instance to get inverse kinematics."
    # NOTE: if robot=Franka, first return is xyz, second is R
    interaction_specs['start_state'] = robot.joint2pose(robot_demo['robot_joint_state'][start_timestep])[0]
    interaction_specs['end_state'] = robot.joint2pose(robot_demo['robot_joint_state'][end_timestep])[0]
    interaction_specs['rotation'] = robot_demo['robot_joint_state'][end_timestep][-2] - robot_demo['robot_joint_state'][start_timestep][-2]

    os.makedirs(f"inter_specs", exist_ok=True)
    cv2.imwrite(f"inter_specs/{time.time()}_start.png", robot_demo['robot_img'][start_timestep])
    cv2.imwrite(f"inter_specs/{time.time()}_end.png", robot_demo['robot_img'][end_timestep])
    print(colored(f'[PlayData] ', 'yellow') + f"Visualizing interaction start state!!")
    return interaction_specs

def display_matched_flow(demo_num: int, inter_num: int,
                         human_flow: np.ndarray,
                         human_obs: np.ndarray,
                         robot_flow: np.ndarray,
                         robot_obs: np.ndarray) -> None:
    # NOTE: This is assuming all absolute flow
    assert human_flow.ndim == 3 and human_flow.shape[-1] == 2, 'Human flow shape should be (T H*W C)'
    assert robot_flow.ndim == 3 and robot_flow.shape[-1] == 2, 'Robot flow shape should be (T H*W C)'
    human_vis = visualize_pred_tracking(human_obs,
                                        einops.rearrange(human_flow, 't n c -> n t c'),
                                        normalized = False
                                        )
    robot_vis = visualize_pred_tracking(robot_obs,
                                         einops.rearrange(robot_flow, 't n c -> n t c'),
                                         normalized = False
                                         )
    combined_vis = np.concatenate((np.array(human_vis), np.array(robot_vis)), axis=2)
    os.makedirs(f"human_flow", exist_ok=True)
    save_to_gif(combined_vis,
                save_path=f"human_flow/{demo_num}_{inter_num}.gif",
                fps = 3)
    return 


@hydra.main(version_base="1.1", config_path="conf", config_name="preprocess_play_data")
def main(cfg: DictConfig) -> None:
    """
    This script preprocesses the play data and human data, both with tracking,
    and aligns them based on the closest action-flow pairs -> *_preset.pkl files.
    """
    robot = hydra.utils.instantiate(cfg.robot)
    # data with tracking
    human_files = natsort.humansorted([f for f in glob.glob(cfg.human_data_path + '/*_tracking.pkl')])
    robot_files = natsort.humansorted([f for f in glob.glob(cfg.robot_data_path + '/*_tracking.pkl')])

    # load human data
    human_data = {}
    for idx, file in enumerate(human_files):
        with open(file, "rb") as f:
            data = pickle.load(f)
        human_data[idx] = {
            'human_img': data['observations'],
            'human_tracking': data['tracking']
        }

    # load play data
    play_data = {}
    for idx, file in enumerate(robot_files):
        with open(file, "rb") as f:
            data = pickle.load(f)
        assert 'type' in data.keys(), "Robot data must have 'type' key for interaction type"
        play_data[idx] = {
            'type': data['type'],
            'robot_img': data['img'],
            'robot_joint_state': data['joint_state'],
            'robot_tracking': data['tracking']
        }

    # Add interaction_specs to play data
    for idx, data in play_data.items():
        play_data[idx]['interaction_specs'] = get_interaction_specs_for_obj(data, robot)

    data = {"img": [],
            "correction_specs": []}
    if cfg.playdata_type == 'preset':
        pbar = tqdm.tqdm(total=len(human_data), desc="Aligning human and play data")
        for demo_num, human_demo in human_data.items():
            demo_tracking = human_demo['human_tracking']
            for inter_num, inter_tracking in enumerate(demo_tracking):
                human_img = human_demo['human_img'][inter_num][0]
                robot_demo_idx, robot_tracking, human_tracking = get_closest_action_flow_pair(inter_tracking, human_img, play_data)
                display_matched_flow(demo_num, inter_num, 
                                    human_tracking, 
                                    human_img,
                                    robot_tracking, 
                                    play_data[robot_demo_idx]['robot_img'][0])

                data['img'].append(human_img)
                data['correction_specs'].append(play_data[robot_demo_idx]['interaction_specs'])
            pbar.update(1)
        pbar.close()
        # with open(os.path.join(cfg.save_path, f"action_preset.pkl"), 'wb') as f:
        #     pickle.dump(data, f)

    elif cfg.playdata_type == 'robot':
        data['flow'] = []
        data['goal_img'] = []
        pbar = tqdm.tqdm(total=len(play_data), desc="Preprocessing robot data")
        for idx in range(len(play_data)):
            data['img'].append(play_data[idx]['robot_img'][0])
            data['goal_img'].append(play_data[idx]['robot_img'][-1])
            data['flow'].append(play_data[idx]['robot_tracking'])
            data['correction_specs'].append(play_data[idx]['interaction_specs'])
            pbar.update(1)
        pbar.close()
        # with open(os.path.join(cfg.save_path, f"playdata.pkl"), 'wb') as f:
        #     pickle.dump(data, f)


    # if cfg.playdata_type == 'preset':
    #     tasks = ["drop_bowl", "pick_block", "sort_screw", "multi_task"]
    #     save_path = '/projects/collab/Human2Robot/EaseScene/robot_video'
    #     play_data = {"img": [],
    #                 "correction_specs": []}
    #     for task in tasks:
    #         with open(os.path.join(save_path, f"{task}/action_preset.pkl"), 'rb') as f:
    #             data = pickle.load(f)
    #         play_data["img"].extend(data["img"])
    #         play_data["correction_specs"].extend(data["correction_specs"])
    #     with open(os.path.join(save_path, f"action_preset.pkl"), 'wb') as f:
    #         pickle.dump(play_data, f)
    # elif cfg.playdata_type == 'robot':
    #     tasks = ["drop_bowl", "pick_block", "sort_screw", "multi_task"]
    #     save_path = '/projects/collab/Human2Robot/EaseScene/robot_video'
    #     play_data = {"img": [],
    #                  "goal_img": [],
    #                 "flow": [],
    #                 "correction_specs": []}
    #     for task in tasks:
    #         with open(os.path.join(save_path, f"{task}/playdata.pkl"), 'rb') as f:
    #             data = pickle.load(f)
    #         play_data["img"].extend(data["img"])
    #         play_data["goal_img"].extend(data["goal_img"])
    #         play_data["flow"].extend(data["flow"])
    #         play_data["correction_specs"].extend(data["correction_specs"])
    #     with open(os.path.join(save_path, f"playdata.pkl"), 'wb') as f:
    #         pickle.dump(play_data, f)


if __name__ == "__main__":
    main()

