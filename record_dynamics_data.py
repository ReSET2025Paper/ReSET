import os
import cv2
import time
import glob
import hydra
import pickle
import numpy as np
from natsort import humansorted
from omegaconf import DictConfig, OmegaConf

from policy import load_policy
from utils import preprocess_video, visualize_video, ObservationBuffer

@hydra.main(version_base=None, config_path='conf', config_name='record_dynamics_data')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.demo_name is None:
        print('Missing save name, here is a list of names')
        print('already saved in the destination folder')
        os.makedirs(cfg.save_path, exist_ok=True)
        print(humansorted(os.listdir(cfg.save_path)))

        demo_name = input("input an unused demo name: ")
    else:
        demo_name = cfg.demo_name

    cameras = {}
    cameras['static_cam'] = hydra.utils.instantiate(cfg.camera.static)
    cameras['gripper_cam'] = hydra.utils.instantiate(cfg.camera.gripper)
    time.sleep(1)

    robot = hydra.utils.instantiate(cfg.robot)
    print('[*] Connecting to robot...')
    conn = robot.connect(8080)
    print('[*] Connection complete')

    print('[*] Connecting to gripper')
    conn_gripper = robot.connect(8081)
    print('[*] Connection complete')

    START = cfg.task.start_position
    robot.send2gripper(conn_gripper, "o")
    robot.go2position(conn, START)
    gripper_state = np.array([0.0])

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
    
    obs = {}
    for key, cam in cameras.items():
        if key == "static_cam":
            obs["image"] = preprocess_video(cam.frame, specs=cfg.video_specs)
        else:
            obs["image_gripper"] = preprocess_video(cam.frame, specs=cfg.video_specs)
    obs['agent_pos'] = robot.readState(conn)
    obs_buffer = ObservationBuffer(buffer_size=4, init_obs=obs)

    mppi_policy = load_policy(
        policy_cfg = cfg.policy,
        robot = robot,
        conn = conn,
        conn_gripper = conn_gripper,
        device = cfg.device,
    ).to(cfg.device)

    # goal = mppi_policy.actions[3]
    # start_state = goal[:3]
    # end_state = goal[3:6]
    # rotation = goal[6]

    # step_time = 1/30
    # start_time = time.time()
    # idx = 0
    # while True:
    #     try:
    #         curr_time = time.time()
    #         if curr_time - start_time >= step_time:
    #             start_time = curr_time
                
    #             obs = {}
                
    #             for key, cam in cameras.items():
    #                 if key == "static_cam": 
    #                     static_frame = cam.frame
    #                     obs["image"] = preprocess_video(static_frame, specs=cfg.video_specs)
    #                 else:
    #                     gripper_frame = cam.frame
    #                     obs["image_gripper"] = preprocess_video(gripper_frame, specs=cfg.video_specs)
    #             robot_state = robot.readState(conn)
    #             # obs['agent_pos'] = robot.readState(conn)
    #             obs['agent_pos'] = robot_state
    #             print(idx, robot_state["q"])
    #             obs_buffer.add(obs)
    #             obs = obs_buffer.get_buffer()


    #             joint_state = obs["agent_pos"][-1]["q"]
    #             curr_cart_state = robot.joint2pose(joint_state)[0]
    #             # print("curr_cart_state: {}".format(curr_cart_state))
    #             # print("curr_goal: {}".format(curr_goal))

    #             dist = np.linalg.norm(curr_cart_state - start_state)   
    #             # xdot = np.clip(curr_goal - curr_cart_state, -0.05, 0.05)
    #             xdot = np.clip(start_state - curr_cart_state, -0.05, 0.05)
    #             xdot = np.concatenate([xdot, np.zeros(3)]) # add rotation to be 0
    #             action = robot.xdot2qdot(xdot, obs["agent_pos"][-1])
    #             robot.send2robot(conn, action, cfg.control_mode)
    #             idx +=1


    #     except KeyboardInterrupt:
    #         print("Recording stopped.")
    #         stop_deploy(cameras, robot, conn, conn_gripper, START)
    #         os.makedirs(cfg.save_path, exist_ok=True)
    #         with open(f"{cfg.save_path}/{demo_name}.pkl", 'wb') as handle:
    #             pickle.dump(trajectory, handle)
    #         break
    # for i in range(200):
    #     print(i)
    #     _ = robot.readState(conn)

    step_time = 1/30
    start_time = time.time()
    idx = 0
    while True:
        try:
            robot_state = robot.readState(conn)
            curr_time = time.time()
            if curr_time - start_time >= step_time:
                print(curr_time - start_time)
                start_time = curr_time
                
                obs = {}
                
                for key, cam in cameras.items():
                    if key == "static_cam": 
                        static_frame = cam.frame
                        obs["image"] = preprocess_video(static_frame, specs=cfg.video_specs)
                    else:
                        gripper_frame = cam.frame
                        obs["image_gripper"] = preprocess_video(gripper_frame, specs=cfg.video_specs)
                
                print(idx, robot_state["q"])
                # obs['agent_pos'] = robot.readState(conn)
                obs['agent_pos'] = robot_state
                obs_buffer.add(obs)
                obs = obs_buffer.get_buffer()

                action = mppi_policy.get_action(obs)
                assert len(action) == 8
                if action[-1] == 1:
                    gripper_state = np.array([1.0])
                    robot.send2gripper(conn_gripper, "c")
                elif action[-1] == 0:
                    gripper_state = np.array([0.0])
                    robot.send2gripper(conn_gripper, "o")
                robot.send2robot(conn, action[:-1], cfg.control_mode)

                assert cameras["static_cam"].frame is not None
                assert cameras["gripper_cam"].frame is not None
                trajectory['img'].append(static_frame)
                trajectory['img_gripper'].append(gripper_frame)
                trajectory['joint_vel'].append(action)
                trajectory['joint_state'].append(obs['agent_pos'][-1]["q"])
                trajectory['gripper_state'].append(action[-1])
                idx += 1 
        except KeyboardInterrupt:
            print("Recording stopped.")
            stop_deploy(cameras, robot, conn, conn_gripper, START)
            os.makedirs(cfg.save_path, exist_ok=True)
            with open(f"{cfg.save_path}/{demo_name}.pkl", 'wb') as handle:
                pickle.dump(trajectory, handle)
            break

def stop_deploy(cameras, robot, conn, conn_gripper, home_pos, visualize_path = None):
    cameras['static_cam'].recording = False
    robot.send2gripper(conn_gripper, "o")
    robot.go2position(conn, home_pos)

    if visualize_path is not None:
        os.makedirs(visualize_path, exist_ok=True)
        file_num = len(glob.glob(f"{visualize_path}/*.mp4"))
        visualize_video(
            video = np.array(cameras['static_cam'].buffer),
            save_path=f"{visualize_path}/video_{file_num}.mp4",
            fps = 80
        )
    return 


if __name__ == "__main__":
    main()

