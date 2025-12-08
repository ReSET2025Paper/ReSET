import cv2
import torch
import numpy as np
import torch.nn.functional as F

def preprocess_video(video: np.ndarray, specs: dict) -> np.ndarray:
    # video of shape T H W C
    image = False
    try:
        assert video.ndim == 4 and video.shape[-1] == 3, "Video should be of shape (T, H, W, C)"
    except AssertionError:
        if video.ndim == 3:
            video = np.expand_dims(video, axis=0)
            image = True
        else:
            raise ValueError("Video should be of shape (T, H, W, C) or (H, W, C) for a single image")
        
    if 'cvtColor' in specs:
        tmp_video = []
        if specs['cvtColor'] == 'RGB2BGR':
            for frame in video: # change this
                tmp_video.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif specs['cvtColor'] == 'BGR2RGB':
            for frame in video:
                tmp_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video = np.array(tmp_video)

    if 'resize' in specs:
        video = torch.tensor(video).permute(0, 3, 1, 2).float()
        video = F.interpolate(video, size=tuple(specs['resize']), mode='bilinear')
        video = video.permute(0, 2, 3, 1).numpy()

    if image:
        video = video.squeeze(0)
    
    return video


def visualize_video(video: np.ndarray, save_path: str, fps: int = 30) -> None:
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