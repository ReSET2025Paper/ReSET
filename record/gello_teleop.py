import glob
import time
import numpy as np

from typing import Tuple
from threading import Thread
from termcolor import colored

from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.agents.gello_agent import GelloAgent

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


class Gello():
    def __init__(self, 
                 port: int = 6001,
                 hostname: str = '127.0.0.1',
                 hz: int = 100,
                 home_pos: np.array = None,
                 ):
        self.robot_client = ZMQClientRobot(port=port, host=hostname)
        self.env = RobotEnv(self.robot_client, control_rate_hz = hz)

        # Assuming GELLO has been connected through USB
        usb_ports = glob.glob("/dev/serial/by-id/*")
        print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"using port {gello_port}")
        else:
            raise ValueError(
                "No gello port found, please specify one or plug in gello"
                )
        self.agent = GelloAgent(port=gello_port)
        self.home_pos = home_pos

    def run(self):
        assert self.agent._robot._torque_on == True, colored('[Gello]', 'red') + 'Make sure gello joints are locked before you start teleoperation!'
        self.obs = None
        self.done = False
        self.gello_thread = Thread(target=self._run)
        self.gello_thread.daemon = True
        self.gello_thread.start()

    def lock_and_check(self):
        ### Lock starting joint
        self._go_home()
        print(colored("[GELLO] ", 'green') + f"Slowly move the joints to match the current robot position.")  
        self.agent._robot.set_torque_mode(False)  
        max_joint_delta = 0.1
        locked = [False, False, False, False, False, False, False]

        while(not all(locked)):
            start_pos = self.agent.act()
            for pos_id in range(len(start_pos)):
                if 6.0 < start_pos[pos_id] < 6.6:
                    start_pos[pos_id] = start_pos[pos_id] - (2 * np.pi)
                if -6.6 < start_pos[pos_id] < -6.0:
                    start_pos[pos_id] = start_pos[pos_id] + (2 * np.pi)

            obs = self.env.get_obs()
            joints = obs["joint_positions"]
            abs_deltas = np.abs(start_pos - joints)

            for id in range(len(locked)):
                if locked[id] == False:
                    if abs_deltas[id] < max_joint_delta:
                        self.agent._robot.set_individual_torque(id + 1)
                        locked[id] = True
                        print(f"Joint {id} locked!")

        if all(locked):
            print(colored("[GELLO] ", 'green') + "All joints locked! Hold still and ready for teleopertation!")
            self.agent._robot._torque_on = True
        return
    
    def _reset(self) -> Tuple[list, list]:
        # going to start position
        print("Going to start position")
        start_pos = self.agent.act()
        obs = self.env.get_obs()
        joints = obs["joint_positions"]

        shifted_neg = []
        shifted_pos = []

        for pos_id in range(len(start_pos)):
            if 6.0 < start_pos[pos_id] < 6.6:
                start_pos[pos_id] = start_pos[pos_id] - (2 * np.pi)
                shifted_neg.append(pos_id)
            if -6.6 < start_pos[pos_id] < -6.0:
                start_pos[pos_id] = start_pos[pos_id] + (2 * np.pi)
                shifted_pos.append(pos_id)

        abs_deltas = np.abs(start_pos - joints)
        id_max_joint_delta = np.argmax(abs_deltas)

        max_joint_delta = 0.8
        if abs_deltas[id_max_joint_delta] > max_joint_delta:
            id_mask = abs_deltas > max_joint_delta
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                ids,
                abs_deltas[id_mask],
                start_pos[id_mask],
                joints[id_mask],
            ):
                print(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            return

        assert len(start_pos) == len(
            joints
        ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

        max_delta = 0.05
        for _ in range(25):
            command_joints = self.agent.act()
            for index in shifted_neg:
                command_joints[index] = command_joints[index] - (2 * np.pi)
            for index in shifted_pos:
                command_joints[index] = command_joints[index] + (2 * np.pi)

            current_joints = obs["joint_positions"]
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            self.env.step(current_joints + delta)

        return shifted_neg, shifted_pos
    
    def _go_home(self) -> None:
        self.env._robot.go_to_position(self.home_pos)
        self.home = True
    
    def _run(self):
        self.home = False
        shifted_neg, shifted_pos = self._reset()
        obs = self.env.get_obs()
        joints = obs["joint_positions"]
        action = self.agent.act()

        for index in shifted_neg:
            action[index] = action[index] - (2 * np.pi)
        for index in shifted_pos:
            action[index] = action[index] + (2 * np.pi)

        if (action - joints > 0.5).any():
            print("Action is too big")

            # print which joints are too big
            joint_index = np.where(action - joints > 0.8)
            for j in joint_index:
                print(
                    f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                )
            exit()
        
        prev_joint_pos = joints
        
        print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))
        start_time = time.time()
        self.agent._robot.set_torque_mode(False)
        timestep = 0
        while not self.done:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            print_color(
                message,
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )
            action = self.agent.act()
            for index in shifted_neg:
                action[index] = action[index] - (2 * np.pi)
            for index in shifted_pos:
                action[index] = action[index] + (2 * np.pi)

            joints = obs["joint_positions"]
            # if (action[:-1] - joints[:-1] > 1.0).any():
            #     print("Action is too big")

            #     # print which joints are too big
            #     joint_index = np.where(action - joints > 1.5)
            #     for j in joint_index:
            #         print(
            #             f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            #         )
            #     exit()

            self.obs = self.env.step(action)
            self.obs["joint_velocities"] = self.obs["joint_positions"] - prev_joint_pos
            self.obs["joint_velocities"][-1] = action[-1]
            self.obs["timestep"] = timestep
            timestep += 1
            prev_joint_pos = self.obs["joint_positions"]

        time.sleep(0.5)
        self._go_home()

if __name__ == "__main__":
    controller = Gello()
    controller.run()
    while True:
        obs = controller.get_obs()
        if obs is not None:
            pass
            # print(obs["joint_velocities"])
            # print(obs["gripper_position"])