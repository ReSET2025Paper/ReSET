import os
import glob
import time
import hydra
import torch
import pickle
import numpy as np
from termcolor import colored
from omegaconf import DictConfig

from utils import visualize_video

def record_human(cfg, demo_num, cam):
    step_time = 0.1
    observations = []
    inter_num = 0
    save_file = f"{cfg.save_path}/demo_{demo_num}.pkl"
    while True: 
        if inter_num > 0:
            print(colored("[Recording] ", "yellow") + f"Press s to start a new intervention, or a to finish demo {demo_num}...")
            time.sleep(0.5)
            while True:
                if cam.key_pressed == 's':
                    time.sleep(0.5)
                    break
                elif cam.key_pressed == 'a':
                    print(colored("[Recording] ", "green") + f"Demo {demo_num} finished with {inter_num} interventions.")
                    dict = {'observations': observations}
                    if cfg.multi_task:
                        dict["text_cond"] = input("Input the task name:")
                    with open(save_file, 'wb') as f:
                        pickle.dump(dict, f)
                    return
            
        print(colored("[Recording] ", "green") + f"Demo {demo_num}, intervention {inter_num} started...")
        print("Press 's' when finishing this intervention.")
        inter_data = []
        start_time = time.time()
        while True:
            curr_time = time.time()
            if curr_time - start_time >= step_time:
                inter_data.append(cam.frame)
                start_time = curr_time
            if cam.key_pressed == 's':
                print(colored("[Recording] ", "green") + f"Intervention {inter_num} finished, {len(inter_data)} observations recorded.")
                break
        inter_data = np.array(inter_data)
        observations.append(inter_data)
        inter_num += 1

        if cfg.save_video:
            os.makedirs(f"{cfg.save_path}/video", exist_ok=True)
            video_path = f"{cfg.save_path}/video/demo_{demo_num}_inter_{inter_num}.mp4"
            visualize_video(inter_data, video_path, fps=30)



@hydra.main(version_base="1.1", config_path="../conf/record", config_name="record_human")
def main(cfg: DictConfig) -> None:
    cam = hydra.utils.instantiate(cfg.camera.static, visualize=True)
    time.sleep(1) 
    os.makedirs(cfg.save_path, exist_ok=True)

    while True:
        all_demo = sorted([demo.split("/")[-1] for demo in glob.glob(cfg.save_path + '/*.pkl')])
        print(colored("[Demos] Existing demos:", "yellow") +  f"{all_demo}")
        demo_num = input(colored("[Demos] Enter demo number to record: ", "yellow"))
        try:
            assert demo_num.isdigit()
        except AssertionError:
            print(colored("[Demos] Invalid input. Please enter a number.", "red"))
            continue

        
        record_human(cfg, demo_num, cam)
        # demo_num += 1
    # cam.done = True
    # print(colored("Recording finished.", "green") + f"{demo_num} demos recorded.")
    # return 

if __name__ == "__main__":
    main()