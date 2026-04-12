import json
import logging
import time
from argparse import ArgumentParser, ArgumentTypeError

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import Server, Trainer, RolloutWorker
from tmrl.tools.check_environment import check_env_tm20lidar, check_env_tm20full
from tmrl.tools.record import record_reward_dist
from tmrl.util import partial
import tmrl

# custom
import random
import numpy as np
import torch
import os
import sys
import platform
from pathlib import Path
from datetime import datetime
import csv
import math


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_metadata(config, seed):
    return {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "tmrl_version": getattr(tmrl, "__version__", "unknown"),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "config": config,
    }


def save_run_metadata_txt(metadata, output_dir, run_name=None):
    os.makedirs(output_dir, exist_ok=True)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"{run_name}_metadata.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, indent=4, default=str))
    return file_path


def patch_training_agent_with_csv(csv_path):
    """Monkey-patch TrainingAgent.train() để ghi loss ra CSV."""
    from tmrl.training import TrainingAgent

    original_train = TrainingAgent.train
    step_counter = [0]
    prev_actor = [None]

    # Khởi tạo CSV với header
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss_critic", "loss_actor",
                                 "actor_delta", "critic_status"])
    logging.info(f"[CSV] Loss logging → {csv_path}")

    def patched_train(self, batch):
        ret = original_train(self, batch)

        step_counter[0] += 1
        step = step_counter[0]
        lc = ret.get("loss_critic", ret.get("loss_q", None))
        la = ret.get("loss_actor", ret.get("loss_pi", None))

        if lc is None or la is None:
            return ret

        # Phát hiện vấn đề
        critic_status = "ok"
        if math.isnan(lc) or math.isinf(lc):
            critic_status = "NaN/Inf"
            logging.warning(f"[STEP {step}] CRITIC LOSS EXPLODED: {lc}")
        elif lc > 1e5:
            critic_status = "exploding"
            logging.warning(f"[STEP {step}] CRITIC LOSS HIGH: {lc:.1f}")

        actor_delta = 0.0
        if prev_actor[0] is not None:
            actor_delta = abs(la - prev_actor[0])
            if actor_delta > 50:
                logging.warning(f"[STEP {step}] ACTOR SPIKE: Δ={actor_delta:.2f}")
        prev_actor[0] = la

        # Ghi CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([step, round(lc, 6), round(la, 6),
                                     round(actor_delta, 4), critic_status])
        return ret

    TrainingAgent.train = patched_train


def main(args):
    set_seed(args.seed)

    run_name = args.run_name if args.run_name else f"seed_{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.server:
        serv = Server()
        while True:
            time.sleep(1.0)

    elif args.worker or args.test or args.benchmark or args.expert:
        config = dict(cfg_obj.CONFIG_DICT)
        config_modifiers = args.config
        for k, v in config_modifiers.items():
            config[k] = v

        metadata = build_run_metadata(config, args.seed)
        saved_path = save_run_metadata_txt(metadata, run_dir, "metadata")
        logging.info(f"Run metadata saved to: {saved_path}")

        rw = RolloutWorker(
            env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),
            actor_module_cls=cfg_obj.POLICY,
            sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
            device='cuda' if cfg.CUDA_INFERENCE else 'cpu',
            server_ip=cfg.SERVER_IP_FOR_WORKER,
            max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
            crc_debug=cfg.CRC_DEBUG,
            standalone=args.test,
            run_dir=run_dir,
            seed=args.seed
        )

        if args.worker:
            rw.run()
        elif args.expert:
            rw.run(expert=True)
        elif args.benchmark:
            rw.run_env_benchmark(nb_steps=1000, test=False)
        else:
            rw.run_episodes(10000)

    elif args.trainer:
        config = dict(cfg_obj.CONFIG_DICT)
        metadata = build_run_metadata(config, args.seed)
        saved_path = save_run_metadata_txt(metadata, run_dir, "metadata")
        logging.info(f"Run metadata saved to: {saved_path}")
        csv_path = os.path.join(run_dir, "loss_log.csv")
        patch_training_agent_with_csv(csv_path)

        trainer = Trainer(
            training_cls=cfg_obj.TRAINER,
            server_ip=cfg.SERVER_IP_FOR_TRAINER,
            model_path=cfg.MODEL_PATH_TRAINER,
            checkpoint_path=cfg.CHECKPOINT_PATH,
            dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,
            load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN,
            updater_fn=cfg_obj.UPDATER_FN,
            run_dir=run_dir
        )

        logging.info(f"--- NOW RUNNING {cfg_obj.ALG_NAME} on TrackMania ---")
        if args.wandb:
            trainer.run_with_wandb(
                entity=cfg.WANDB_ENTITY,
                project=cfg.WANDB_PROJECT,
                run_id=cfg.WANDB_RUN_ID
            )
        else:
            trainer.run()

    elif args.record_reward:
        record_reward_dist(path_reward=cfg.REWARD_PATH, use_keyboard=args.use_keyboard)

    elif args.check_env:
        if cfg.PRAGMA_LIDAR:
            check_env_tm20lidar()
        else:
            check_env_tm20full()

    elif args.install:
        logging.info(f"TMRL folder: {cfg.TMRL_FOLDER}")

    else:
        raise ArgumentTypeError('Enter a valid argument')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--install', action='store_true', help='checks TMRL installation')
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--expert', action='store_true', help='launches an expert rollout worker (no model update)')
    parser.add_argument('--test', action='store_true', help='runs inference without training')
    parser.add_argument('--benchmark', action='store_true', help='runs a benchmark of the environment')
    parser.add_argument('--record-reward', dest='record_reward', action='store_true', help='utility to record a reward function in TM20')
    parser.add_argument('--use-keyboard', dest='use_keyboard', action='store_true', help='modifier for --record-reward')
    parser.add_argument('--check-environment', dest='check_env', action='store_true', help='utility to check the environment')
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='(use with --trainer) if you want to log results on Weights and Biases, use this option')
    parser.add_argument('-d', '--config', type=json.loads, default={}, help='dictionary containing configuration options (modifiers) for the rtgym environment')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--run-name', type=str, default='First', help='optional run name')
    parser.add_argument('--output-dir', type=str, default=r'C:\Users\Admin\TmrlData', help='directory for run artifacts')
    arguments = parser.parse_args()

    main(arguments)