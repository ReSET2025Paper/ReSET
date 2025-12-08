import os
import cv2
import time
import hydra
import pickle
import numpy as np
from natsort import os_sorted
from omegaconf import DictConfig, OmegaConf

from gello_teleop import Gello


@hydra.main(version_base=None, config_path='../conf/record', config_name='record_robot')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.demo_name is None:
        print('Missing save name, here is a list of names')
        print('already saved in the destination folder')
        os.makedirs(cfg.save_path, exist_ok=True)
        print(os_sorted(os.listdir(cfg.save_path)))

        demo_name = input("input an unused demo name: ")
    else:
        demo_name = cfg.demo_name


    static_cam = hydra.utils.instantiate(cfg.camera.static)
    gripper_cam = hydra.utils.instantiate(cfg.camera.gripper)
    time.sleep(1)

    # Home position
    
    START = cfg.task.middle_position if (cfg.type == "execution" and "middle_position" in cfg.task) else cfg.task.start_position
    START.append(0.0)

    interface = Gello(home_pos=START)

    # Data to be saved
    trajectory = {'joint_state': [],
                  'ee_state': [],
                  'gripper_state': [],
                  'joint_vel': [],
                  'ee_vel': [],
                  'gripper_action': [],
                  'img': [],
                  'img_gripper': [],
                  }
    if cfg.type == 'correction':
        trajectory['type'] = input("Interaction type (pull/pick/rotate): ")
    
    interface.lock_and_check()
    run = True
    record = False
    # step_time = 0.02
    step_time = 1 / 30 

    print("[*] Press A to start recording demos")

    while True:
        if static_cam.key_pressed == 'a':
            record = True
            start_time = time.time()
            prev_time = time.time()
            interface.run()
            print("[*] Recording start...")
            break

    while run:

        if static_cam.key_pressed == 's':
            run = False
            # print(f"This demo has: {len(trajectory['joint_state'])} data points")
            # print("[*] Press A to to save the recorded demo")
            time.sleep(0.01)

        # if static_cam.key_pressed == 'a':
        #     run = False

        # Update demo if recording 
        curr_time = time.time()
        if record and curr_time - prev_time >= step_time:
            assert static_cam.frame is not None
            assert gripper_cam.frame is not None
            obs = interface.obs
            if obs is not None:
                trajectory['img'].append(static_cam.frame)
                trajectory['img_gripper'].append(gripper_cam.frame)
                trajectory['joint_vel'].append(obs['joint_velocities'])
                trajectory['joint_state'].append(obs['joint_positions'])
                trajectory['gripper_action'].append(obs['joint_velocities'][-1])

            prev_time = curr_time

    # Kill camera threads
    static_cam.done = True
    gripper_cam.done = True
    interface.done = True

    print(f"This demo has: {len(trajectory['joint_vel'])} data points")
    print(f"Data recording is at {len(trajectory['joint_vel']) / (time.time() - start_time)}")
    os.makedirs(cfg.save_path, exist_ok=True)
    with open(f"{cfg.save_path}/{demo_name}.pkl", "wb") as handle:
        pickle.dump(trajectory, handle)

    while not interface.home:
        pass
    return

if __name__== '__main__':
    main()