## Check if rl model is work
Runtime data/config folder: C:\Users\Admin\TmrlData 
Command line Interface: D:\Internship-AutoDrivingCar\tmrl\docs\source\cli.rst

# Install the tmrl-test map (first run)
Copy:
    C:\Users\Admin\TmrlData\resources\tmrl-test.Map.Gbx
to:
    C:\Users\Admin\OneDrive\Documents\Trackmania\Maps\My Maps

# Prepare Trackmania window exactly like TMRL expects
Open trackmania game -> create -> track editor -> edit a map -> tmrl-test -> Launch the tmrl-test track -> hitting the green flag.
Hide the ghost with G

# Start / reload the TMRL OpenPlanet plugin
In Trackmania:
    Press F3 (OpenPlanet overlay)
    Go Developer → Load plugin / Reload plugin → “TMRL grab data”
    Then open OpenPlanet(in OpenPlanet overlay) → Log and confirm see something like “Waiting for incoming connection…”

# Run a pre-trained agent 

In terminal: python -m tmrl --test      
    -> confirm: screenshots + telemetry + virtual gamepad control are all working.
NOTE: your lap will become LAGGY and FAN is LOUDING

## Train my own agent
https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md at 
    Train your own self-driving AI
    tmrl enables training your own policies, on your own tracks:

Open 3 terminals:
    python -m tmrl --server
    python -m tmrl --check-environment
    python -m tmrl --trainer
    python -m tmrl --worker
Baseline: (TM20IMAGES)
    python -m tmrl --server
    python -m tmrl --worker --seed 0 --run-name baseline_seed1
    python -m tmrl --trainer --seed 0 --run-name baseline_seed1
Baseline: (TM20LIDAR)
    python -m tmrl --server
    python -m tmrl --worker --seed 0 --run-name SAC_lidar_v1_seed1
    python -m tmrl --trainer --seed 0 --run-name SAC_lidar_v1_seed1
    LINUX
        python3 -m tmrl --server
            INFO:2026-04-07 09:23:34,238:Local for server: Started to connect.
            INFO:2026-04-07 09:23:34,238:Listening on TCP to port 6666
            INFO:2026-04-07 09:23:34,239:Local for server: Connected.
        python3 -m tmrl --trainer --seed 0 --run-name SAC_lidar_v1_seed0

Troubleshooting (the 2 most common blockers)
    No connection / nothing happens when you run --test
    Make sure the OpenPlanet plugin is actually loaded (F3 → Developer → Reload “TMRL grab data”) and watch OpenPlanet logs.
    Port conflict (connection fails)
    TMRL uses a fixed port by default; if another app is using it, you’ll see connection issues. A maintainer notes this commonly happens when something is already using the port (e.g., 9000).

“Avoidance system” in Trackmania: RL vs CV
In Trackmania, “avoidance” usually means don’t hit walls / don’t go off-track / don’t get stuck. RL is typically the most effective because the agent learns control (steer/throttle/brake), not just detection.



## TMRL engine (of theirs)
Path to edit configurations for model training: "C:\Users\Admin\TmrlData\config\config.json"

# Reward setup: BASED on theory of target
What is Our target ? (what should i choose)
    Safety
    Efficiency (lowest time to finish the road)
    Progress/Task completion (complete the road)
    Comfortable 
    Follow the rules of env ?
    Realibility: handle many situation 

    NOTE: Because in Trackmania, the priorities
        Most important
            finish the track
            minimize completion time
            avoid crashes that lose time
            stay on the valid route
        Still useful
            smooth control
            stable recovery after mistakes
        Less important than real-road driving
            traffic laws
            passenger comfort

Setup reward value (for each correct action/wrong action)
    there is no single “best” reward config for every map. In TMRL, the base reward is progress along a recorded demo trajectory, and the config values mainly shape when an episode ends, how tolerant the system is to shortcuts/straying, and how strongly finishing is favored
    Experience: tune by yourself


## Car state while driving 

## Runtime + reward score per epoch saving in CSV file 
C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\site-packages\tmrl\networking.py 
    *Have modified to save file in .csv, IF run python -m tmrl --worker inside PS D:\Internship-AutoDrivingCar>, .csv file will be in this folder path

## TMNF
C:\Users\Admin\OneDrive\Documents\TMInterface\Plugins\Python_Link.as 


## Problems: know and avoid
# Setup/pipeline 
No samples arrive at the trainer.
Wrong reward trajectory for the map.
Window/camera mismatch.
Plugin or communication dropouts.

# RL / model-training problems
Sample inefficiency.
Sparse or badly shaped rewards
Reward hacking / unintended shortcuts.
Crashing into walls or getting stuck.
Poor exploration.
Training instability.
Overfitting to one map or one trajectory.
Weak interpretability.

# Good signs vs bad signs
Good signs:
    worker repeatedly collects episodes and sends buffers
    trainer stops only waiting and begins updating
    average episode reward trends upward over time
    finish rate improves
    average runtime to finish drops
Bad signs:
    trainer always waits for samples
    worker times out or disconnects
    reward stays flat or becomes worse
    episodes end quickly with lots of crashes/stuck behavior
    the car learns weird shortcuts instead of valid racing lines


