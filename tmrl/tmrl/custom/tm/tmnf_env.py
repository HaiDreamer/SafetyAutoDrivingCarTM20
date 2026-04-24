from rtgym import DEFAULT_CONFIG_DICT, RealTimeGym
from tmnf_interfaces import TMNFInterface  # your file

# Copy the default config and override what you need
MY_CONFIG = DEFAULT_CONFIG_DICT.copy()
MY_CONFIG["interface"] = TMNFInterface
MY_CONFIG["time_step_duration"] = 0.05        # seconds per step (matches on_step_period)
MY_CONFIG["start_obs_capture"] = 0.04
MY_CONFIG["time_step_timeout_factor"] = 1.0
MY_CONFIG["ep_max_length"] = 1000
MY_CONFIG["act_buf_len"] = 1                  # no action buffer needed for discrete
MY_CONFIG["reset_act_buf"] = True
MY_CONFIG["interface_kwargs"] = {"game_speed": 2.0, "port": 8477}

import gymnasium
gymnasium.register(
    id="tmnf-v0",
    entry_point=lambda: RealTimeGym(config=MY_CONFIG),
)