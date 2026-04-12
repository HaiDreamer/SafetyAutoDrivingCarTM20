All question in 
    config_object ?

Exploration: the car tries a slightly different steering angle, braking point, or racing line.
Exploitation: the car repeats the line that already gave high reward.

Why have both trainer and worker ?

# Update log file
finished = go to the end of map
truncated = only when max_samples hit 
runtime_sec can includes the reset/sleep time at the start of each episode (cfg.SLEEP_TIME_AT_RESET which is typically 1-2 seconds)
terminated = Car reached the finish line (data[8] from OpenPlanet) OR reward function triggered failure (strayed too far, too many zero-reward steps)

# SAC alogrithm
The Big Picture: 3 Processes Running in Parallel
┌─────────────┐     samples      ┌─────────────┐     weights     ┌─────────────┐
│   Worker    │ ──────────────►  │   Server    │ ◄────────────── │   Trainer   │
│ (plays TM)  │                  │ (relay hub) │ ──────────────► │ (runs SAC)  │
└─────────────┘                  └─────────────┘                 └─────────────┘
The server consolidates model weights and passes them to the trainer, while the trainer is responsible for actually training. The worker is responsible for interacting with the game. The trainer runs SAC in a loop on batches sampled from the replay memory — a large buffer of past (observation, action, reward, next_observation, done) transitions collected by workers.

NetworkRoleactor
The policy — takes observation, outputs a probability distribution over actions (steering, gas, brake)
q1, q2Twin critics — each takes (observation, action) and outputs a Q-value. Two used to prevent overestimation
q1_target, q2_target - Frozen slow-moving copies of critics, used to compute stable training targets

train() loop
    each training step does this with a batch (o, a, r, o2, d) — observation, action, reward, next observation, done
    Compute the critic target (what Q should be)
    Update the critics: loss_q = MSE(q1(o, a), backup) + MSE(q2(o, a), backup)
    Update the actor

Why SAC is Powerful
    Advantages:
     High sample efficiency (off-policy)
     Stable training (entropy + double Q)
     Works well in continuous control
     Less hyperparameter tuning
    Weaknesses:
     More complex than DQN / PPO
     Computationally heavier
     Sensitive to reward scaling

# Fix window-> linux model 
(tmrl_env) student5@ict16:~$ cat > ~/patch_pathlib.py << 'PYEOF'
with open('/home/student5/miniconda3/envs/tmrl_env/lib/python3.13/pathlib/_local.py', 'r') as f:
    content = f.read()

old = "    if os.name != 'nt':\n        def __new__(cls, *args, **kwargs):\n            raise UnsupportedOperation(\n                f\"cannot instantiate {cls.__name__!r} on your system\")"

new = "    if False:\n        def __new__(cls, *args, **kwargs):\n            raise UnsupportedOperation(\n                f\"cannot instantiate {cls.__name__!r} on your system\")"

if old in content:
    content = content.replace(old, new)
    with open('/home/student5/miniconda3/envs/tmrl_env/lib/python3.13/pathlib/_local.py', 'w') as f:
        f.write(content)
    print('Patched successfully')
else:
    print('Pattern not found - no changes made')
PYEOF
python ~/patch_pathlib.py
Patched successfully
(tmrl_env) student5@ict16:~$ python ~/fix_checkpoint.py
Done! Fixed checkpoint saved to: /home/student5/TmrlData/checkpoints/SAC_lidar_v1_t.tcpt



# custom_models.py
Model           | Observation Type             | Backbone            | Notes
----------------|------------------------------|---------------------|-----------------------------
MLP             | Float only                   | Linear layers       | Simplest
REDQ MLP        | Float only                   | Linear layers       | 10 Q-networks
EfficientNet    | Images + floats              | EfficientNetV2-S    | Heavy, accurate
Vanilla CNN     | Grayscale images + floats    | 4-layer CNN         | Lightweight
Color CNN       | RGB images + floats          | Same CNN            | Discards color anyway
RNN             | Float sequences              | GRU + MLP           | Unsupported


# What model best and why (pls find citation later)
Algorithm | Action Space | On/Off Policy | Best For                          | TrackMania Fit
----------|--------------|---------------|-----------------------------------|----------------
SAC       | Continuous   | Off-policy    | Continuous control, sample efficiency | ⭐⭐⭐⭐⭐
TD3       | Continuous   | Off-policy    | Stable continuous control         | ⭐⭐⭐⭐
DDPG      | Continuous   | Off-policy    | Continuous, but unstable          | ⭐⭐⭐
PPO       | Both         | On-policy     | Stability, robotics               | ⭐⭐⭐
A3C       | Both         | On-policy     | Parallel environments             | ⭐⭐
DQN       | Discrete     | Off-policy    | Discrete only                     | ⭐

Why SAC wins for TrackMania
1. Continuous action space is natural
Steering, gas, brake are all analog values — SAC handles this natively without discretizing. DQN is out for this reason alone.
2. Entropy maximization = better exploration
SAC adds an entropy term to its objective, encouraging the agent to explore diverse driving lines rather than collapsing to one mediocre policy early. This is critical in racing where there are many valid trajectories.
3. Off-policy = sample efficient
SAC reuses past experience via a replay buffer. In tmrl, collecting real-time data from TrackMania is slow — you can't afford to throw away transitions like on-policy methods (PPO, A3C) do.
4. More stable than DDPG/TD3
DDPG is notorious for instability. TD3 fixes many of DDPG's issues and is actually a close second, but SAC's entropy regularization gives it an extra robustness edge in high-variance environments like racing.

Why not the others?
    PPO — Good algorithm, but on-policy means low sample efficiency. Every time you update, old data is discarded. Fine for simulations that run at 1000x speed, but tmrl runs in real-time.
    A3C — Same on-policy problem, plus async workers add complexity without major gains here.
    DDPG — Predecessor to both TD3 and SAC. Strictly dominated by both.
    DQN — Only works with discrete actions. You'd need to manually bucket steering angles, losing precision.

TL;DR
    SAC is the best choice for tmrl-style racing because it handles continuous control natively, is sample-efficient (off-policy)(IS THAT TRUE WITH tmrl IMPLEMENT?), and explores better thanks to entropy regularization. TD3 is a solid second if you want a simpler/more deterministic policy.


## Q-value
Q-value comes from Q-learning. It answers the question:
    "If I'm in state S and take action A, how much total future reward will I get?"
More formally:
    Q(s, a) = immediate reward + discounted sum of all future rewards
        s: Current observation — speed, LIDAR distances, history of past frames
        a: Action chosen — gas, brake, steering angle
        Q(s, a): Expected total future reward (e.g., lap completion speed) if you take action a now and then follow the best policy afterward
Why SAC has two critics (twin Q-networks):
    SAC takes the minimum of two independent Q-value estimates to avoid overestimation bias. This is why self.output_layers in PopArt accepts a list — both critic heads get normalized together with the same mean/std.

