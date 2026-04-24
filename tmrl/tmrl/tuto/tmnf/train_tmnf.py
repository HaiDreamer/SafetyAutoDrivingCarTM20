# train_tmnf.py
import tmrl.config.config_constants as cfg
from tmrl.networking import Server, RolloutWorker, Trainer
from tmrl.training import TrainingOffline
from tmrl.training_offline import TorchTrainingOffline
from tmrl.actor import TorchActorModule
from tmrl.util import partial

from tmrl.custom.tm import tmnf_env

# ── Actor: reuse tmrl's default MLP actor ─────────────────────────────
from tmrl.custom.custom_models import Mlp1ActorModule  # simple MLP

# ── Server ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    role = sys.argv[1]  # "server", "worker", or "trainer"

    if role == "server":
        server = Server()
        while True:
            server.run()  # blocks

    elif role == "worker":
        worker = RolloutWorker(
            env_id="tmnf-v0",
            actor_module_cls=Mlp1ActorModule,
            sample_compressor=None,
            device="cpu",
            server_ip=cfg.SERVER_IP_FOR_WORKER,
            min_samples_per_worker_packet=1,
            max_samples_per_episode=1000,
        )
        worker.run()

    elif role == "trainer":
        trainer = Trainer(
            training_cls=TorchTrainingOffline,
            server_ip=cfg.SERVER_IP_FOR_TRAINER,
            device="cuda",  # or "cpu"
        )
        trainer.run()