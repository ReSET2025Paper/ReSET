import os
import cv2
import math
import glob
import einops
import pickle
import natsort
import numpy as np

from termcolor import colored
from scipy.interpolate import interp1d
from policy import CorrectionPolicy
from utils import visualize_pred_tracking, write_to_frame, save_to_gif

class PresetOpenloopPolicy(CorrectionPolicy):
    def __init__(self, 
                 preset_path: str,
                 ** kwags) -> None:
        super().__init__(
            **kwags)
        """
        Policy that matches predited flow with a preset of action-flow pairs.
        action_flow_pair: dict{"intervention_num":
                                                {"action": np.ndarray, 
                                                 "flow": np.ndarray}
                                }
        """
        self._get_action_flow_preset(preset_path)
        self.curr_action_sequence = None
        self.curr_action_index = 0

    def get_action(self, obs_dict: dict[str, np.array]) -> np.ndarray: # obs shape: (B, C, H, W), normalized
        obs = obs_dict['image'][-1]
        if self.curr_action_sequence is None or self.curr_action_index >= len(self.curr_action_sequence):
            flow, uncertainty = self._get_flow_and_uncertainty(obs)
            print(colored("[PresetOpenloop] ", "green") + f"Current uncertainty is: {uncertainty}")
            self.curr_action_index = 0
            if self.visualize_path is not None:
                os.makedirs(self.visualize_path, exist_ok=True)
                write_frame = write_to_frame(obs.copy(), f"{uncertainty:.3f}")
                num_files = len(glob.glob(f"{self.visualize_path}/*.png"))
                cv2.imwrite(f"{self.visualize_path}/uncertainty_{num_files}.png", write_frame)
            if uncertainty > self.uncertainty_threshold:
                self.curr_action_sequence, closest_flow = self._get_closest_action_flow_pair(flow) #input is a relative flow
                if self.visualize_path is not None:
                    self._display_matched_flow(flow, closest_flow, obs)
                    print(colored("[PresetOpenloop] ", "green") + f"Uncertainty and matched flow saved to {self.visualize_path}...")
            else:
                return None, None

        action = self.curr_action_sequence[self.curr_action_index] * self.action_scale
        self.curr_action_index += 1
        last_action_for_intervention = (self.curr_action_index == len(self.curr_action_sequence) - 1)

        return action, last_action_for_intervention

    def _get_closest_action_flow_pair(self, flow_obs: np.ndarray) -> tuple:
        closest_dist = float('inf')
        closest_action = None
        closest_flow = None
        assert flow_obs.ndim == 4 and flow_obs.shape[0] == 2, 'Flow shape should be (C T H W)'
        flow_obs = einops.rearrange(flow_obs, 'c t h w -> t (h w) c') # relative, normalized

        t_flow_obs = np.linspace(0, 1, flow_obs.shape[0] + 1) # + 1 Given the flow from robot demos are absolute

        for intervention_num, pair in self.action_flow_pairs.items():
            action = pair['action']
            flow = pair['flow'] # T N C

            # Dowsampling the flow to match the observation flow
            t_flow_sample = np.linspace(0, 1, flow.shape[0])
            N = flow.shape[1]  # number of actions
            flow_downsampled = []
            for n in range(N):
                interp_x = interp1d(t_flow_sample, flow[:, n, 0], kind='linear', fill_value='extrapolate')
                interp_y = interp1d(t_flow_sample, flow[:, n, 1], kind='linear', fill_value='extrapolate')
                x_new = interp_x(t_flow_obs)
                y_new = interp_y(t_flow_obs)
                flow_downsampled.append(np.stack([x_new, y_new], axis=-1))  # shape: (T_obs, 2)
            flow_downsampled = np.stack(flow_downsampled, axis=1)  # shape: (T_obs, N, 2)

            # convert the flow to relative flow
            relative_flow = flow_downsampled[1:, :, :] - flow_downsampled[:-1, :, :]
            relative_flow = np.where(np.abs(relative_flow) < 1, np.zeros_like(relative_flow), relative_flow)
            min_max = np.concatenate([self.min_flow, self.max_flow], axis=0)
            bound = abs(min_max).max(axis=0)
            relative_flow = (relative_flow / bound) * 100

            dist = np.linalg.norm(flow_obs - relative_flow)
            if dist < closest_dist:
                closest_dist = dist
                closest_action = action
                closest_flow = flow_downsampled
        
        closest_flow = einops.rearrange(closest_flow, 't (h w) c-> c t h w', h = int(math.sqrt(closest_flow.shape[1])))

        return closest_action, closest_flow
    
    def _get_action_flow_preset(self, preset_path: str) -> None:
        print(colored("[PresetOpenloop] ", "green") + f"Loading action-flow pairs from {preset_path}...")
        demos = natsort.humansorted(glob.glob(preset_path + '/*.pkl'))
        demos = [demo for demo in demos if 'tracking' in demo]

        self.action_flow_pairs = {}
        for idx, demo in enumerate(demos):
            with open(demo, 'rb') as f:
                data = pickle.load(f)
            self.action_flow_pairs[idx] = {
                'action': data['joint_vel'],  
                'flow': data['tracking']  
            }
        return
    
    def _display_matched_flow(self, flow_obs: np.ndarray, 
                              flow_pred: np.ndarray, 
                              obs: np.ndarray) -> None:
        # convert flows to C T H W
        assert flow_obs.ndim == 4 and flow_pred.ndim == 4, "Flows are in wrong shape"
        obs_flow = visualize_pred_tracking(
                                obs,
                                flow_obs,
                                relative_flow=True,
                                denoise_flow=True,
                                max_flow=self.max_flow,
                                min_flow=self.min_flow,)
        
        matching_flow = visualize_pred_tracking(
                                obs,
                                flow_pred,
                                normalized=False)
        
        combined_flow = np.concatenate((np.array(obs_flow), np.array(matching_flow)), axis=2)  # concatenate along width

        # display = True
        # while display:
        #     for image in combined_flow:
        #         for i in range(50):
        #             cv2.imshow("Flow Prediction", image / 255)
        #             cv2.waitKey(1)

        if self.visualize_path is not None:
            file_num = len(glob.glob(f"{self.visualize_path}/*.gif"))
            save_to_gif(
                combined_flow,
                save_path=f"{self.visualize_path}/matched_flow_{file_num}.gif",
                fps = 3
            )
            

# def main():
#     preset_path = '/projects/collab/Human2Robot/EaseScene/robot_video/pull_bowl/correction'
#     policy = PresetOpenloopPolicy(preset_path=preset_path, device='cuda:0',
#                           uncertainty_threshold=0.5,)

# if __name__ == "__main__":
#     main()
    