import threading
import socket
import struct
import time
import numpy as np
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from rtgym import RealTimeGymInterface
import gymnasium.spaces as spaces
import tmrl.config.config_constants as cfg

# ── Message type constants (must match Python_Link.as enum) ──────────────────
SC_RUN_STEP_SYNC               = 1
SC_CHECKPOINT_COUNT_CHANGED    = 2
SC_LAP_COUNT_CHANGED           = 3
SC_REQUESTED_FRAME_SYNC        = 4
SC_ON_CONNECT_SYNC             = 5
C_SET_SPEED                    = 6
C_REWIND_TO_STATE              = 7
C_REWIND_TO_CURRENT_STATE      = 8
C_GET_SIMULATION_STATE         = 9
C_SET_INPUT_STATE              = 10
C_GIVE_UP                      = 11
C_PREVENT_SIMULATION_FINISH    = 12
C_SHUTDOWN                     = 13
C_EXECUTE_COMMAND              = 14
C_SET_TIMEOUT                  = 15
C_RACE_FINISHED                = 16
C_REQUEST_FRAME                = 17
C_RESET_CAMERA                 = 18
C_SET_ON_STEP_PERIOD           = 19
C_UNREQUEST_FRAME              = 20
C_TOGGLE_INTERFACE             = 21
C_IS_IN_MENUS                  = 22
C_GET_INPUTS                   = 23

TMI_PORT = 8477
LIDAR_SIZE = 19


# ── Low-level binary helpers ─────────────────────────────────────────────────

def _recv_exactly(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf

def _send_int32(sock, value):
    sock.sendall(struct.pack('<i', value))

def _send_float(sock, value):
    sock.sendall(struct.pack('<f', value))

def _send_uint8(sock, value):
    sock.sendall(struct.pack('<B', value))

def _recv_int32(sock):
    return struct.unpack('<i', _recv_exactly(sock, 4))[0]

def _recv_float(sock):
    return struct.unpack('<f', _recv_exactly(sock, 4))[0]


# ── TMInterface socket client ────────────────────────────────────────────────

class TMNFClient:
    def __init__(self, port=TMI_PORT, game_speed=2.0, save_replays=False):
        self.port = port
        self.game_speed = game_speed

        self.race_time = 0
        self.checkpoint_count = 0
        self.checkpoint_target = 0
        self.race_finished = False
        self._lock = threading.Lock()

        self._sock = None
        self._connected = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._connected.wait(timeout=30)

        self.speed = 0.0
        self.position = (0.0, 0.0, 0.0)
        self.lidar = [0.0] * LIDAR_SIZE

    def _run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', self.port))
        server.listen(1)
        print(f"[TMNFClient] Waiting for TMInterface on port {self.port}...")
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = conn
        print(f"[TMNFClient] Connected from {addr}")
        self._handle_on_connect()
        self._connected.set()
        self._event_loop()

    def _handle_on_connect(self):
        msg_type = _recv_int32(self._sock)
        assert msg_type == SC_ON_CONNECT_SYNC, f"Expected SC_ON_CONNECT_SYNC, got {msg_type}"
        self.set_speed(self.game_speed)
        _send_int32(self._sock, SC_ON_CONNECT_SYNC)

    def _event_loop(self):
        while True:
            try:
                msg_type = _recv_int32(self._sock)
            except Exception:
                print("[TMNFClient] Connection lost")
                break

            if msg_type == SC_RUN_STEP_SYNC:
                race_time = _recv_int32(self._sock)
                speed     = _recv_float(self._sock)
                pos_x     = _recv_float(self._sock)
                pos_y     = _recv_float(self._sock)
                pos_z     = _recv_float(self._sock)
                lidar     = [_recv_float(self._sock) for _ in range(LIDAR_SIZE)]
                with self._lock:
                    self.race_time = race_time
                    self.speed     = speed
                    self.position  = (pos_x, pos_y, pos_z)
                    self.lidar     = lidar
                _send_int32(self._sock, SC_RUN_STEP_SYNC)

            elif msg_type == SC_CHECKPOINT_COUNT_CHANGED:
                current = _recv_int32(self._sock)
                target  = _recv_int32(self._sock)
                with self._lock:
                    self.checkpoint_count = current
                    self.checkpoint_target = target
                    if current >= target:
                        self.race_finished = True
                _send_int32(self._sock, SC_CHECKPOINT_COUNT_CHANGED)

            elif msg_type == SC_LAP_COUNT_CHANGED:
                current = _recv_int32(self._sock)
                target  = _recv_int32(self._sock)
                _send_int32(self._sock, SC_LAP_COUNT_CHANGED)

            elif msg_type == SC_REQUESTED_FRAME_SYNC:
                _send_int32(self._sock, SC_REQUESTED_FRAME_SYNC)

            else:
                print(f"[TMNFClient] Unknown message type: {msg_type}")

    def set_speed(self, speed: float):
        _send_int32(self._sock, C_SET_SPEED)
        _send_float(self._sock, speed)

    def set_input_state(self, left=False, right=False, accelerate=False, brake=False):
        _send_int32(self._sock, C_SET_INPUT_STATE)
        _send_uint8(self._sock, int(left))
        _send_uint8(self._sock, int(right))
        _send_uint8(self._sock, int(accelerate))
        _send_uint8(self._sock, int(brake))

    def give_up(self):
        _send_int32(self._sock, C_GIVE_UP)

    def get_state(self):
        with self._lock:
            return (
                self.race_time,
                self.checkpoint_count,
                self.checkpoint_target,
                self.race_finished,
            )

    def reset_race_finished(self):
        with self._lock:
            self.race_finished = False
            self.checkpoint_count = 0


# ── TMRL RealTimeGymInterface ────────────────────────────────────────────────

class TMNFInterface(RealTimeGymInterface):
    def __init__(self, game_speed=2.0, port=TMI_PORT, save_replays=False):
        self.client = TMNFClient(port=port, game_speed=game_speed)
        self.window = WindowInterface("TrackMania")
        self.prev_cp = 0
        self._last_race_time = 0
        self.reward_function = RewardFunction(
            reward_data_path=cfg.REWARD_PATH,
            nb_obs_forward=cfg.REWARD_CONFIG['CHECK_FORWARD'],
            nb_obs_backward=cfg.REWARD_CONFIG['CHECK_BACKWARD'],
            nb_zero_rew_before_failure=cfg.REWARD_CONFIG['FAILURE_COUNTDOWN'],
            min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],
            max_dist_from_traj=cfg.REWARD_CONFIG['MAX_STRAY'],
        )

    def get_observation_space(self):
        return spaces.Tuple((
            spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32),
            spaces.Box(low=0.0, high=1.0,    shape=(LIDAR_SIZE,), dtype=np.float32),
        ))

    def get_action_space(self):
        return spaces.MultiBinary(4)

    def get_default_action(self):
        return np.array([0, 0, 1, 0], dtype=np.int8)

    def send_control(self, control):
        left, right, accelerate, brake = [bool(x) for x in control]
        self.client.set_input_state(
            left=left,
            right=right,
            accelerate=accelerate,
            brake=brake,
        )

    def reset(self, seed=None, options=None):
        self.client.give_up()
        self.client.reset_race_finished()
        self.reward_function.reset()
        self.prev_cp = 0
        self._last_race_time = 0
        # wait until the game actually restarts
        timeout = time.time() + 10
        while self.client.race_time > 100 and time.time() < timeout:
            time.sleep(0.05)
        obs = self._build_obs()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        race_time, cp, cp_target, finished = self.client.get_state()
        obs = self._build_obs()

        pos = np.array(self.client.position, dtype=np.float32)
        reward, terminated = self.reward_function.compute_reward(pos)

        if finished:
            terminated = True

        self._last_race_time = race_time
        info = {"race_time": race_time, "checkpoint": cp, "cp_target": cp_target}
        return obs, reward, terminated, info

    def wait(self):
        pass

    def _build_obs(self):
        speed = np.array([self.client.speed], dtype=np.float32)
        lidar = np.array(self.client.lidar, dtype=np.float32)
        return (speed, lidar)