import time
import pickle
import numpy as np

from . import Policy

class OpenLoop(Policy):
    """
    Open-loop policy that returns a fixed action sequence.
    """

    def __init__(self, action_path: str = None, 
                 **kwags):
        super().__init__(**kwags)

        with open(action_path, 'rb') as f:
            data = pickle.load(f)
        self.action_sequence = data["joint_vel"]
        self.states = data['joint_state']
        # self.action_sequence = self.skip_sampling(self.states, 1)
        self.curr_action_index = 0
        self.start_time = None

    def get_action(self, obs_dict: dict[str, np.array] = None) -> np.ndarray:
        # step_time = 0.08 #0.08
        if self.curr_action_index >= len(self.action_sequence):
            return None  
        # if self.start_time is None:
        #     self.start_time = time.time()
        # curr_time = time.time()
        # print(curr_time - self.start_time)

        action = self.action_sequence[self.curr_action_index] * self.action_scale
        # if curr_time - self.start_time >= step_time:
        self.curr_action_index += 1
            # self.start_time = curr_time

        return action
    
    def skip_sampling(self, states: np.array, skip_frame: int) -> np.ndarray:
        # int = 0
        # joint_vel = []
        # while int + skip_frame < len(states):
        #     vel = states[int + skip_frame] - states[int]
        #     joint_vel.append(vel)
        #     int += skip_frame

        # get vel from states
        prev_states = np.array(states)[:-1, :-1]
        next_states = np.array(states)[1:, :-1]
        gripper_states = np.array(self.action_sequence)[1:,-1]
        joint_vel = next_states - prev_states

        # remove zero values
        remove_inst = []
        for i in range(len(joint_vel)):
            if np.linalg.norm(joint_vel[i]) == 0:
                remove_inst.append(i)
        gripper_states = np.expand_dims(gripper_states, axis = 1)
        joint_vel = np.concatenate((joint_vel, gripper_states), axis = 1)
        joint_vel = np.delete(joint_vel, remove_inst, axis=0)

        # skip frames
        skipped_joint_vel = []
        int = 0 
        while int + skip_frame < len(joint_vel):
            target = joint_vel[int: int + skip_frame + 1]
            skipped_joint_vel.append(np.sum(list(target), axis = 0))
            int = int + skip_frame

        skipped_joint_vel = [i for i in skipped_joint_vel for _ in range(5)]
        # return np.array(joint_vel)
        return np.array(skipped_joint_vel)
    


