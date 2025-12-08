"""
Calls and runs the PI0 policy from:
@article{black2410pi0,
  title={$\pi$0: A vision-language-action flow model for general robot control. CoRR, abs/2410.24164, 2024. doi: 10.48550},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint ARXIV.2410.24164}
}
"""
import time
import msgpack
import asyncio
import functools
import threading
import websockets
import numpy as np
from termcolor import colored
from . import Policy
from utils import preprocess_video

def _ws_is_open(ws) -> bool:
  state = getattr(ws, "state", None)
  if state is not None:
      return "OPEN" in str(state)

  closed = getattr(ws, "closed", None)
  if closed is not None:
      return not closed
  
  return getattr(ws, "close_code", None) is None

def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj

def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj

Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class PI0(Policy):
  def __init__(self, 
              name: str,
              prompt: str, 
              **kwags) -> None:
    super(PI0, self).__init__(**kwags)
    # This should be connecting to the server after running 'uv run scripts/serve_policy.py'
    # Also run 'ssh -J daiyinlong@falcon1.arc.vt.edu -N -L 8000:10.128.38.49:8000 daiyinlong@{node}' to forward the port
    # Also run 'ssh -J {user_name}@{server_address} -N -L 8000:{ip}:8000 {user_name}@{node}' to forward the port
    self.uri = f"ws://127.0.0.1:8000"
    self._reconnect = True
    self._connect_timeout = 60.0
    self.prompt = prompt
    self.action_buffer = []

    # ---- NEW: shared state for background inference ----
    self._latest_obs = None
    self._obs_lock = threading.Lock()
    self._obs_event = threading.Event()
    self._act_lock = threading.Lock()
    self._loop_thread = None
    # ----------------------------------------------------

    try:
        self._loop = asyncio.get_event_loop()
        if self._loop.is_closed():
            raise RuntimeError
    except Exception:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    self._packer = Packer(use_bin_type=True)
    # self._ws = self._loop.run_until_complete(self._connect_and_handshake())
    # print(colored("[PI0] ", "green") + f"Connected to policy server at {self.uri}")

    # ---- CHANGED: run loop in background thread and start inferer ----
    self._start_loop_thread()                   # NEW
    fut = asyncio.run_coroutine_threadsafe(self._connect_and_handshake(), self._loop)
    self._ws = fut.result()
    print(colored("[PI0] ", "green") + f"Connected to policy server at {self.uri}")
    # kick off background inference forever
    asyncio.run_coroutine_threadsafe(self._infer_forever(), self._loop)  # NEW
    # -----------------------------------------------------------------

  # ---- NEW: spin the event loop in a daemon thread ----
  def _start_loop_thread(self):
    if self._loop.is_running():
        return
    def _runner():
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    self._loop_thread = threading.Thread(target=_runner, daemon=True)
    self._loop_thread.start()
  # -----------------------------------------------------

  async def _connect_and_handshake(self):
    ws = await asyncio.wait_for(
        websockets.connect(
            self.uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ),
        timeout=self._connect_timeout,
    )

    first = await ws.recv() # first meta data
    self._server_metadata = unpackb(first, raw=False)
    return ws

  async def _ensure_conn(self):
      if self._ws is None or not _ws_is_open(self._ws):
          if not self._reconnect:
              raise ConnectionError("WebSocket is closed and reconnect is disabled.")
          self._ws = await self._connect_and_handshake()
  
  async def _infer_async(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    await self._ensure_conn()

    obs = self._convert2le(obs_dict)
    await self._ws.send(self._packer.pack(obs))
    raw = await self._ws.recv()
    resp = unpackb(raw, raw=False)

    if "actions" not in resp:
        raise KeyError(f"Response missing 'actions' key: {resp.keys()}")
    actions = np.asarray(resp["actions"], dtype=np.float32)
    print(f"actions:{actions.shape}")
    print(resp["server_timing"])
    return actions

  # ---- NEW: background loop that always uses newest obs and flushes buffer ----
  import time

  async def _infer_forever(self):
      last_infer_time = 0.0
      while True:
          # wait until someone posts a new obs
          while not self._obs_event.is_set():
              await asyncio.sleep(0.001)

          # grab the newest obs and clear the flag
          with self._obs_lock:
              obs = self._latest_obs
              self._latest_obs = None
          self._obs_event.clear()
          if obs is None:
              await asyncio.sleep(0)  # spurious wake-up
              continue

          now = time.time()
          if now - last_infer_time < 1 / 30:
              await asyncio.sleep(0.001)
              continue

          # run inference, with simple reconnect handling
          try:
              actions = await self._infer_async(obs)
          except websockets.ConnectionClosed:
              self._ws = None
              try:
                  actions = await self._infer_async(obs)
              except Exception:
                  await asyncio.sleep(0.001)
                  continue
          except Exception:
              await asyncio.sleep(0.001)
              continue

          # flush & replace buffer with fresh actions (append gripper here)
        #   processed = [np.append(a, 0.0).astype(np.float32) for a in actions]
          processed = np.array([a for a in actions], dtype=np.float32)
          with self._act_lock:
              self.action_buffer.clear()
              self.action_buffer.extend(processed)

          last_infer_time = now
  # ---------------------------------------------------------------------------

#   def get_action(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
#     if len(self.action_buffer) == 0:
#         try:
#             action = self._loop.run_until_complete(self._infer_async(obs_dict))
#             for a in action:
#                 # a = np.append(a, 0.0)  # add gripper state
#                 # a[-1] = 0.0
#                 self.action_buffer.append(a)
#             action = self.action_buffer.pop(0)
#             action[:-1] = action[:-1] * 30
#             return action
#         except websockets.ConnectionClosed:
#             self._ws = None
#             return self._loop.run_until_complete(self._infer_async(obs_dict))
#     else:
#         action = self.action_buffer.pop(0)
#         action[:-1] = action[:-1] * 30
#         return action

  def get_action(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    with self._obs_lock:
      self._latest_obs = obs_dict
    self._obs_event.set()

    # serve from buffer; if empty, wait a tiny bit for fresh result (optional)
    for _ in range(2):  # small, bounded wait (~2ms total if loop tick is 1ms)
      with self._act_lock:
        if self.action_buffer:
          action = self.action_buffer.pop(0)
          action[:-1] = action[:-1] * 3
          return action
      time.sleep(0.001)
    
    print("buffer empty!")

    return np.zeros(8, dtype=np.float32)
  
  def _convert2le(self, obs_dict: dict[str, np.ndarray]) -> dict:
    # print(f"remaining actions{self.prompt}")
    # print(obs_dict['image'].shape)
    image = preprocess_video(np.squeeze(obs_dict['image'][-1]), {"cvtColor": 'BGR2RGB'})
    wrist_image = preprocess_video(np.squeeze(obs_dict['image_gripper'][-1]), {"cvtColor": 'BGR2RGB'})
    # image = np.squeeze(obs_dict['image'][-1])
    # wrist_image = np.squeeze(obs_dict['image_gripper'][-1])
    return {
      "observation/image": image.astype(np.uint8),
      "observation/wrist_image": wrist_image.astype(np.uint8),
      "observation/state": np.squeeze(np.asarray(obs_dict['agent_pos'], dtype=np.float32)[:,-1]),
      "prompt": self.prompt,
      } 

