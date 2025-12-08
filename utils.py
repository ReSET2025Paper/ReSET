import cv2
import math
import time
import torch
import socket
import einops
import pygame
import imageio
import colorsys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import cm
from threading import Thread
from typing import Optional, Tuple

############################################## Vision Utils ####################################################
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

def preprocess_video(video: np.array, specs: dict) -> np.array:
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

def denoise_flow(relative_flow: np.array, threshold: float = 1.0) -> np.array:
    """
    Denoising relative flow by thresholding
    """
    # denoise on (x,y) l2 norm
    norm = np.linalg.norm(relative_flow, axis=-1, keepdims=True)
    relative_flow = np.where(norm > threshold, relative_flow, 0.0)
    return relative_flow


def preprocess_flow(flow: np.array, 
                    normalized: bool = True,
                    relative: bool = True,
                    denoise: bool = True,
                    H: int = 20,
                    W: int = 20,
                    min_flow: float = 0.0, # placeholder
                    max_flow: float = 1.0 # placeholder
                    ) -> np.array:
    # Input flow likely to be an output of a flow predictor
    # flow shape: N T C
    T = flow.shape[1]

    if normalized:
        if relative:
            # # denormalize from [-1, 1]
            # min_max = np.concatenate([min_flow, max_flow], axis=0)
            # bound = abs(min_max).max(axis=0)
            # bound = 300 # NOTE: check if the bound is valid
            # flow = flow / 500 * bound
            flow = flow
        else:
            # denormalize from [0,1]
            # flow = flow * (max_flow - min_flow + 1e-8)  +  min_flow  
            flow = flow

    if relative:
        if denoise:
            flow = denoise_flow(flow)
        grid = get_points_on_a_grid(int(math.sqrt(flow.shape[0])), (H, W))
        grid = einops.rearrange(np.repeat(grid, T, axis = 0), 'T N C -> N T C')
        abs_flow = np.cumsum(flow, axis=1)  
        flow = grid + abs_flow 
        flow = np.concatenate([grid[:, :1], flow], axis=1)

    return flow
    

def visualize_pred_tracking(init_obs: np.array,
                            pred_tracking: np.array, # N, T, C (2) or (C, T, H, W)
                            normalized: bool = True,
                            relative_flow: bool = False,
                            denoise_flow: bool = False,
                            min_flow: float = 0.0, # placeholder
                            max_flow: float = 1.0, # placeholder
                            viz_horizon: int = 10,
                            save_fps: int = 3,
                            save_path: str = None) -> np.array:
    """
    Unlike the defualt visualize method from CoTracker, this one visualzes tracking traj on ONE initial frame
    """


    try:
        assert pred_tracking.ndim == 3, "pred_tracking should be of shape (N, T, C)"
    except AssertionError:
        if pred_tracking.ndim == 4 and pred_tracking.shape[-1] ==  pred_tracking.shape[-2]:
            pred_tracking = einops.rearrange(pred_tracking, 'C T H W -> (H W) T C')
        else:
            raise ValueError("pred_tracking should be of shape (N, T, C) or (C, T, H, W)")
        
    try:
        assert init_obs.ndim == 3 and init_obs.shape[-1] == 3, "init_obs should be of shape (H, W, C)"
    except AssertionError:
        if init_obs.ndim == 3 and init_obs.shape[0] == 3:
            init_obs = einops.rearrange(init_obs, 'C H W -> H W C')
        else:
            raise ValueError("init_obs should be of shape (H, W, C) or (C, H, W)")
    
    N = pred_tracking.shape[0]
    H, W = init_obs.shape[:2]

    pred_tracking = preprocess_flow(
        flow = pred_tracking,
        normalized = normalized,
        relative = relative_flow,
        denoise = denoise_flow,
        H = H, W = W,
        min_flow = min_flow,
        max_flow = max_flow
    )
    disp_thresh = 20.0
    displacements = []
    for n in range(N):
        traj = pred_tracking[n]  # shape [T, 2]
        if traj.shape[0] < 2:
            displacements.append(0.0)
            continue
        diffs = np.diff(traj, axis=0)
        total_disp = np.linalg.norm(diffs, axis=1).sum()
        displacements.append(total_disp)
    displacements = np.array(displacements)


    viz_gif = []
    for i in range(1,pred_tracking.shape[1]):
        vis_points = pred_tracking[:, max(0, i + 1 - viz_horizon) : i + 1, :]
        frame = init_obs.copy()
        for n in range(N):
            if displacements[n] < disp_thresh: 
                continue

            color_map = cm.get_cmap("jet")
            color = np.array(color_map(n / max(1, float(N - 1)))[:3]) * 255
            
            color_alpha = 1
            hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
            color = colorsys.hsv_to_rgb(hsv[0], hsv[1] * color_alpha, hsv[2])
            
            for t in range(vis_points.shape[1] - 1):
                start_point = (int(vis_points[n][t][0]), int(vis_points[n][t][1]))
                end_point = (int(vis_points[n][t + 1][0]), int(vis_points[n][t + 1][1]))
                cv2.line(frame, 
                        start_point, end_point, 
                        color=color,
                        thickness=1,
                        lineType=16)
                if t == len(vis_points[n]) - 2:
                    cv2.circle(frame, 
                            end_point, 
                            radius=3, 
                            color=color, 
                            thickness=-1, 
                            lineType=16)
        viz_gif.append(frame)
    
    if save_path is not None:
        save_to_gif(viz_gif, save_path if save_path else "pred_tracking.gif", fps=save_fps)
    return viz_gif



def save_to_gif(frames: list, save_path: str, fps: int = 30, bbox: np.ndarray = None) -> None:
    processed_frames = []
    for frame in frames:
        assert frame.ndim == 3, "Each frame should be of shape (H, W, C)"
        assert frame.shape[-1] == 3, "Each frame should have 3 channels (RGB)"

        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        processed_frames.append(frame)

    imageio.mimsave(
        save_path, 
        processed_frames, 
        format='GIF', 
        duration=1/fps
    )

def write_to_frame(frame: np.array,
                   text: str,
                   position: tuple = (10, 30),
                   font_scale: float = 1,
                   color: tuple = (255, 255, 255),
                   thickness: int = 2) -> np.array:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame

"""
from co-tracker repo:
@inproceedings{karaev24cotracker3,
  title     = {CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos},
  author    = {Nikita Karaev and Iurii Makarov and Jianyuan Wang and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  booktitle = {Proc. {arXiv:2410.11831}},
  year      = {2024}
}
"""
def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


########################################## Model Utils ##################################################

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def set_seed(seed: int = None) -> None:
    """
    Set the random seed for reproducibility.
    """
    seed = seed if seed is not None else 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

############################################# Robot Utils ####################################################

class FrankaResearch3(object):

    def __init__(self):
        self.home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

    def connect(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('172.16.0.3', port))
        s.listen()
        conn, addr = s.accept()
        return conn

    def send2gripper(self, conn, command):
        send_msg = "s," + command + ","
        conn.send(send_msg.encode())

    def send2robot(self, conn, qdot, control_mode, limit=1.0):
        qdot = np.asarray(qdot)
        scale = np.linalg.norm(qdot)
        if scale > limit:
            qdot *= limit/scale
        send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
        if send_msg == '0.,0.,0.,0.,0.,0.,0.':
            send_msg = '0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000'
        send_msg = "s," + send_msg + "," + control_mode + ","
        conn.send(send_msg.encode())

    def listen2robot(self, conn):
        state_length = 7 + 42
        message = str(conn.recv(2048))[2:-2]
        state_str = list(message.split(","))
        for idx in range(len(state_str)):
            if state_str[idx] == "s":
                state_str = state_str[idx+1:idx+1+state_length]
                break
        try:
            state_vector = [float(item) for item in state_str]
        except ValueError:
            return None
        if len(state_vector) is not state_length:
            return None
        state_vector = np.asarray(state_vector)
        states = {}
        states["q"] = state_vector[0:7]
        states["J"] = state_vector[7:49].reshape((7,6)).T
        return states

    def readState(self, conn):
        while True:
            states = self.listen2robot(conn)
            if states is not None:
                break
        return states

    def xdot2qdot(self, xdot, states):
        J_inv = np.linalg.pinv(states["J"])
        return J_inv @ np.asarray(xdot)
    
    def qdot2xdot(self, qdot, states):
        return states["J"] @ np.asarray(qdot)

    def joint2pose(self, q):
        def RotX(q):
            return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
        def RotZ(q):
            return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        def TransX(q, x, y, z):
            return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
        def TransZ(q, x, y, z):
            return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        H1 = TransZ(q[0], 0, 0, 0.333)
        H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
        H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
        H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
        H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
        H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
        H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
        H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
        T = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
        R = T[:,:3][:3]
        xyz = T[:,3][:3]
        return xyz, R

    def go2position(self, conn, goal=False, total_time=15.0):
        if not goal:
            goal = self.home
        start_time = time.time()
        states = self.readState(conn)
        dist = np.linalg.norm(states["q"] - goal)
        elapsed_time = time.time() - start_time
        print(dist)
        while dist > 0.01 and elapsed_time < total_time:
            qdot = np.clip(goal - states["q"], -0.1, 0.1)
            self.send2robot(conn, qdot, "v")
            states = self.readState(conn)
            dist = np.linalg.norm(states["q"] - goal)
            elapsed_time = time.time() - start_time

    def go2cartesian(self, conn, cart_goal=None):
        assert cart_goal is not None, "Goal in catesian must be provided"
        total_time = 15.0
        start_time = time.time()
        states = self.readState(conn)
        curr_cart = self.joint2pose(states["q"])[0]
        dist = np.linalg.norm(curr_cart - cart_goal)
        elapsed_time = time.time() - start_time
        while dist > 0.02 and elapsed_time < total_time:
            xdot = np.clip(cart_goal - curr_cart, -0.05, 0.05)
            xdot = np.concatenate([xdot, np.zeros(3)]) # add rotation to be 0
            qdot = self.xdot2qdot(xdot, states)
            self.send2robot(conn, qdot, "v")
            states = self.readState(conn)
            curr_cart = self.joint2pose(states["q"])[0]
            dist = np.linalg.norm(curr_cart - cart_goal)
            elapsed_time = time.time() - start_time

##################################################### Deployment Utils ###################################################

class ObservationBuffer:
    def __init__(self, 
                 buffer_size: int = 4,
                 init_obs: dict = None) -> None:
        self.buffer_size = buffer_size
        self.buffer = {k: [v]*buffer_size for k, v in init_obs.items()}

    def add(self, obs_dict:dict) -> None:
        for k, v in self.buffer.items():
            v.pop(0)
            v.append(obs_dict[k])

    def get_buffer(self) -> list:
        return {k: np.array(v) for k, v in self.buffer.items()}

    def clear(self):
        self.buffer = {}