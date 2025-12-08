import os
import time
import glob
import hydra
import numpy as np
from termcolor import colored
from omegaconf import DictConfig

from utils import preprocess_video, visualize_video, ObservationBuffer
from policy import load_policy

def deploy(cfg) -> None:

    cameras = {}
    cameras["static_cam"] = hydra.utils.instantiate(cfg.camera.static)
    if "gripper" in cfg.camera:
        cameras["gripper_cam"] = hydra.utils.instantiate(cfg.camera.gripper)

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

    print(colored("[Pipeline] ", "green") + f'Deploy pipeline initinialized.')
    input("Enter to start ...")

    obs = {}
    for key, cam in cameras.items():
        if key == "static_cam":
            obs["image"] = preprocess_video(cam.frame, specs=cfg.video_specs)
        else:
            obs["image_gripper"] = preprocess_video(cam.frame, specs=cfg.video_specs)
    obs['agent_pos'] = np.concatenate((robot.readState(conn)["q"], gripper_state))
    obs_buffer = ObservationBuffer(buffer_size=4, init_obs=obs)

    cameras['static_cam'].recording = True
    # correction policy
    if "correction_policy" in cfg.policy:
        correction_policy = load_policy(
            policy_cfg = cfg.task.load_policy[cfg.policy.correction_policy],
            device = cfg.device,
            robot = robot,
            conn = conn,
            conn_gripper = conn_gripper,
            flow_model_path = cfg.task.flow_model_path,
            uncertainty_model_path = cfg.task.uncertainty_model_path,
            uncertainty_threshold = cfg.task.uncertainty_threshold
        ).to(cfg.device)
        step_time = 1 / 30
        start_time = time.time()
        while True:
            try:
                curr_time = time.time()
                if curr_time - start_time >= step_time:
                    start_time = curr_time

                    obs = {}
                    for key, cam in cameras.items():
                        if key == "static_cam":
                            obs["image"] = preprocess_video(cam.frame, specs=cfg.video_specs)
                        else:
                            obs["image_gripper"] = preprocess_video(cam.frame, specs=cfg.video_specs)
                    obs['agent_pos'] = np.concatenate((robot.readState(conn)["q"], gripper_state))
                    obs_buffer.add(obs)
                    obs = obs_buffer.get_buffer()

                    action, last_action = correction_policy.get_action(obs)

                    if action is None:
                        print("Correction done!")
                        break

                    assert len(action) == 8
                    if action[-1] >= 1.0:
                        gripper_action = "c"
                        gripper_state = np.array([1.0])
                    else:
                        gripper_action = "o"
                        gripper_state = np.array([0.0])
                    robot.send2gripper(conn_gripper, gripper_action)
                    robot.send2robot(conn, action[:-1], cfg.control_mode)

                    if last_action:
                        print("last_action!")
                        time.sleep(2)
                        robot.send2gripper(conn_gripper, "o")
                        robot.go2position(conn, START)
                        gripper_state = np.array([0.0])

            except KeyboardInterrupt:
                stop_deploy(cameras, robot, conn, conn_gripper, START, cfg.task.visualize_path)
                return
    if "middle_position" in cfg.task:
        robot.go2position(conn, cfg.task.middle_position)

    # main policy
    policy = load_policy(
        policy_cfg = cfg.task.load_policy[cfg.policy.base_policy],
        device = cfg.device,
        action_scale = cfg.action_scale
    ).to(cfg.device)
    step_time = 1 / 30
    start_time = time.time()
    gripper_action = "o"
    while True:
        try:
            curr_time = time.time()
            if curr_time - start_time >= step_time:
                start_time = curr_time

                obs = {}
                for key, cam in cameras.items():
                    if key == "static_cam":
                        obs["image"] = preprocess_video(cam.frame, specs=cfg.video_specs)
                    else:
                        obs["image_gripper"] = preprocess_video(cam.frame, specs=cfg.video_specs)
                obs['agent_pos'] = np.concatenate((robot.readState(conn)["q"], gripper_state))
                obs_buffer.add(obs)
                obs = obs_buffer.get_buffer()

                action = policy.get_action(obs)
                print(f"action: {action[-1]}")
                if action is None:
                    print("Policy done, exiting")
                    break

                assert len(action) == 8
                # if action[-1] >= 0.4: 
                if action[-1] >= 0.5:
                    gripper_state = np.array([1.0])
                    robot.send2gripper(conn_gripper, "c")
                    if gripper_action == "o":
                        time.sleep(1)
                    gripper_action = "c"
                elif action[-1] <= 0.2:
                    gripper_action = "o"
                    gripper_state = np.array([0.0])
                    robot.send2gripper(conn_gripper, gripper_action)
                robot.send2robot(conn, action[:-1], cfg.control_mode)   

        except KeyboardInterrupt:
            stop_deploy(cameras, robot, conn, conn_gripper, START, cfg.task.visualize_path)
            break
                    
    return

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

@hydra.main(version_base="1.1", config_path="conf", config_name="deploy")
def main(cfg: DictConfig) -> None:

    deploy(cfg)

if __name__ == "__main__":
    main()