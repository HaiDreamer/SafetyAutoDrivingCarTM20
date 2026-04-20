# rtgym interfaces for Trackmania

# standard library imports
import logging
import time
from collections import deque

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

# third-party imports
from rtgym import RealTimeGymInterface

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_mouse import mouse_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost

r'''
Interfaces for Bridge the game and RL framework so the agent can send action, real observation, cal reward, reset episodes
    Action applied 
    What does the agent observe
    what reward does it get
    Is the episode over
action space = agent được phép điều khiển những gì, theo kiểu dữ liệu nào, trong khoảng giá trị nào -> send to env trackmania 
telemetry là các số liệu game-time: speed, position, has finished?, .....
observation space: [speed, gear, rpm, image history]
'''


# Globals ==============================================================================================================

CHECK_FORWARD = 500  # this allows (and rewards) 50m cuts
# _lidar_log = open("lidar_calibration.txt", "w", buffering=1)  # line-buffered

# Interface for Trackmania 2020 ========================================================================================

class TM2020Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control TrackMania 2020
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 gamepad: bool = True,
                 save_replays: bool = False,
                 grayscale: bool = True,
                 resize_to=(64, 64)):
        """
        Base rtgym interface for TrackMania 2020 (Full environment)

        Args:
            img_hist_len: int: history of images that are part of observations
            gamepad: bool: whether to use a virtual gamepad for control
            save_replays: bool: whether to save TrackMania replays on successful episodes
            grayscale: bool: whether to output grayscale images or color images
            resize_to: Tuple[int, int]: resize output images to this (width, height)
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.small_window = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = cfg.REWARD_CONFIG['END_OF_TRACK']
        self.constant_penalty = cfg.REWARD_CONFIG['CONSTANT_PENALTY']

        self.initialized = False

    def initialize_common(self):
        # Create a virtual gamepad if one is enabled, so the agent can send controller inputs.
        self._last_action = self.get_default_action()
        if self.gamepad:
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
            logging.debug(" virtual joystick in use")
        # Find the Trackmania window and place it in the expected position/size.
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        # Store the current time as the initialization reference point.
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        # Initialize the reward function with trajectory and failure settings.
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,
                                              nb_obs_forward=cfg.REWARD_CONFIG['CHECK_FORWARD'],
                                              nb_obs_backward=cfg.REWARD_CONFIG['CHECK_BACKWARD'],
                                              nb_zero_rew_before_failure=cfg.REWARD_CONFIG['FAILURE_COUNTDOWN'],
                                              min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],
                                              max_dist_from_traj=cfg.REWARD_CONFIG['MAX_STRAY'])
        # Create the OpenPlanet client to read live telemetry from Trackmania.
        self.client = TM2020OpenPlanetClient()

    def initialize(self):
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        """
        Use virtual gamepad if enabled; otherwise send keyboard controls on the laptop.
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [forward,backward,right,left]
        """
        self._last_action = control if control is not None else self.get_default_action()  
        if self.gamepad:
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0.2:           # gas: only if clearly positive
                    actions.append('f')
                elif control[1] > 0.2:         # brake: only if clearly positive AND not gassing
                    actions.append('b')
                # both near zero → coast (neither appended)
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions)

    def grab_data_and_img(self):
        '''take screen shot form Trackmaniaand pull telemetry from the game client'''
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.client.retrieve_data()
        self.img = img  # for render()
        return data, img

    def reset_race(self):
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)  # must be long enough for image to be refreshed

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        self.reward_function.reset()
        return obs, {}

    def close_finish_pop_up_tm20(self):
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        if self.save_replays:
            save_ghost()
            time.sleep(1.0)
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        rew, terminated = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]]),
            speed=float(data[0]),
            action=self._last_action
        )
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            terminated = True
            rew += self.finish_reward
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))  # cv2 images are (h, w, c)
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, save_replays: bool = False):
        super().__init__(img_hist_len, gamepad, save_replays)
        self.window_interface = None
        self.lidar = None
        self.last_crash_time = 0.0        # timestamp of last counted crash
        self.crash_debounce_sec = 1.0     # minimum seconds between crash counts
        self.crash_count = 0
        self.wall_hit_threshold = 5.0    # lidar distance below this = near wall
        self.wall_penalty = 0.5          # (positive) base magnitude — scaled by speed below
        self.soft_zone = 20.0            # proximity gradient starts here
        self.wall_hugging_penalty = 0.05 # per-step penalty while inside soft_zone
        self.was_near_wall = False
        self.max_crashes = 20            # if collide too much, terminate

    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([
            data[0],
        ], dtype='float32')
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        self.crash_count = 0
        self.was_near_wall = False
        self.last_crash_time = 0.0 
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        # count as crash if reward_function terminates due to no progress (car stuck on wall)
        if terminated and not bool(data[8]):  # not end_of_track = stuck/failure termination
            now = time.time()
            if now - self.last_crash_time >= self.crash_debounce_sec:
                self.crash_count += 1
                self.last_crash_time = now
                speed_factor = float(speed[0]) / 100.0
                rew -= self.wall_penalty * (1.0 + speed_factor)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = {}

        # --- WALL COLLISION DETECTION via LIDAR ---
        min_lidar = float(np.min(img))
        current_lidar = img[-1]  # only the most recent frame, shape (19,)
        min_lidar_nonzero = current_lidar[current_lidar > 0]
        min_nonzero = float(np.min(min_lidar_nonzero)) if len(min_lidar_nonzero) > 0 else 999.0
        near_wall = min_nonzero < self.wall_hit_threshold

        # 1. Proximity gradient: small continuous penalty scaling from soft_zone down to wall_hit_threshold
        if min_nonzero < self.soft_zone:
            t = (self.soft_zone - min_nonzero) / (self.soft_zone - self.wall_hit_threshold)
            rew -= self.wall_hugging_penalty * float(np.clip(t, 0.0, 1.0))

        # 2. Crash event: debounced, speed-scaled hard penalty
        if near_wall:
            now = time.time()
            if now - self.last_crash_time >= self.crash_debounce_sec:
                self.crash_count += 1
                self.last_crash_time = now
                speed_factor = float(speed[0]) / 100.0
                rew -= self.wall_penalty * (1.0 + speed_factor)   # e.g. -1.0 at 100 km/h
            if self.crash_count >= self.max_crashes:
                terminated = True
                # print(f"[DEBUG] Terminated by crash_count={self.crash_count}, min_nonzero={min_nonzero:.1f}")  # remove later
                print(f"[DEBUG] near_wall={near_wall}, min_nonzero={min_nonzero:.1f}, terminated={terminated}, crash_count={self.crash_count}")
            self.was_near_wall = True
        else:
            self.was_near_wall = False

        info = {
            "crash_count": self.crash_count,
            "min_lidar": min_lidar,
            "near_wall": near_wall,
            "finished": end_of_track,
            "crashed_out": self.crash_count >= self.max_crashes,
        }
        # ----------------------------------------

        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, imgs))


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        progress = np.array([0], dtype='float32')
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        # count as crash if reward_function terminates due to no progress (car stuck on wall)
        if terminated and not bool(data[8]):  # not end_of_track = stuck/failure termination
            now = time.time()
            if now - self.last_crash_time >= self.crash_debounce_sec:
                self.crash_count += 1
                self.last_crash_time = now
                speed_factor = float(speed[0]) / 100.0
                rew -= self.wall_penalty * (1.0 + speed_factor)
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, progress, imgs]
        end_of_track = bool(data[8])
        info = {}

        # --- WALL COLLISION DETECTION via LIDAR ---
        current_lidar = img[-1]  # only the most recent frame, shape (19,)
        min_lidar_nonzero = current_lidar[current_lidar > 0]
        min_nonzero = float(np.min(min_lidar_nonzero)) if len(min_lidar_nonzero) > 0 else 999.0
        near_wall = min_nonzero < self.wall_hit_threshold

        # Proximity gradient
        if min_nonzero < self.soft_zone:
            t = (self.soft_zone - min_nonzero) / (self.soft_zone - self.wall_hit_threshold)
            rew -= self.wall_hugging_penalty * float(np.clip(t, 0.0, 1.0))

        # Crash event: debounced, speed-scaled hard penalty
        if near_wall:
            now = time.time()
            if now - self.last_crash_time >= self.crash_debounce_sec:
                self.crash_count += 1
                self.last_crash_time = now
                speed_factor = float(speed[0]) / 100.0
                rew -= self.wall_penalty * (1.0 + speed_factor)
            if self.crash_count >= self.max_crashes:
                terminated = True
                # print(f"[DEBUG] Terminated by crash_count={self.crash_count}, min_nonzero={min_nonzero:.1f}")  # remove later
                print(f"[DEBUG] near_wall={near_wall}, min_nonzero={min_nonzero:.1f}, terminated={terminated}, crash_count={self.crash_count}")
        # ----------------------------------------

        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, progress, imgs))


if __name__ == "__main__":
    pass