import cv2
import time
import torch
import pygame
import numpy as np
from threading import Thread

# NOTE: This code should be identical to EaseSCene/utils.py

class camera():
    def __init__(self, cam_id, visualize=False):

        self.cap = cv2.VideoCapture(cam_id)
        time.sleep(0.5)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.done = False
        self.frame = None
       
        self.cam_id = cam_id
        self.visualize = visualize
        self.buffer = []
        self.recording = False

        self.frame_thread = Thread(target=self._run)
        self.frame_thread.daemon = True
        self.frame_thread.start()

    def get_frame(self):
        _, frame = self.cap.read()
        return frame
    
    def _run(self):
        if self.visualize:
            display = pygame.display.set_mode((640, 480))

        while not self.done:
            self.frame = self.get_frame()
            if self.recording:
                self.buffer.append(self.frame)
            
            if self.visualize:
                # cv2.imshow(str(self.cam_id), self.frame)
                # cv2.waitKey(1)

                image = np.rot90(self.frame)
                image = np.flipud(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(image)
                display.blit(surf, (0, 0))
                pygame.display.flip()
                self.key_pressed = None
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_a:
                            self.key_pressed = 'a'
                        elif event.key == pygame.K_s:
                            self.key_pressed = 's'
                        elif event.key == pygame.K_v:
                            self.key_pressed = 'v'
                        elif event.key == pygame.K_y:
                            self.key_pressed = 'y'
                        elif event.key == pygame.K_n:
                            self.key_pressed = 'n'

def visualize_video(video: np.array, save_path: str, fps: int = 30) -> None:
    # video of shape T H W C
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video = video.astype(np.uint8)
    try:
        assert video.ndim == 4 and video.shape[-1] == 3, "Video should be of shape (T, H, W, C)"
    except AssertionError:
        video = torch.tensor(video)
        if video.dim() == 5:
            video = video.squeeze(0)
        if video.shape[-1] != 3:
            video = video.permute(0, 2, 3, 1)
        video = np.array(video, dtype = np.uint8)    
    assert video.ndim == 4 and video.shape[-1] == 3, "Video should be of shape (T, H, W, C)"
    
    height, width, _ = video[0].shape
    video_length = video.shape[0]
    frame_size = (width, height)
    vid_w = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    
    for frame_num in range(video_length):
        vid_w.write(video[frame_num])
    vid_w.release()
    print(f"video saved at {save_path}")