"""Base of image algorithms (INPUT = "images"): a network trained over rounds on raw images.

ImageServer builds the network from the model section (flcore/models/network.py), keeps the global
state, sends it to the selected clients, averages what they return (FedAvg, weights = training samples)
and evaluates the global network on the test images. Subclasses are algorithms: FedAvg and FedAvgJoint
(serverfedavg.py) use it as it is; another algorithm overrides aggregate(), its client's loss(), or adds
hooks, and keeps the shared loop of serverbase.py.

Messages are dicts {"state": {name: tensor}} (download) and {"state": ..., "n": samples} (upload), counted
by the byte rule of comm_utils (4 bytes per float). The state holds the trainable tensors (and batch-norm
statistics of trained modules); the head travels for the classes seen so far only.

torch is imported when the module loads if it is installed; without it, listings, settings checks and
dry runs still work, and init_algorithm() stops with an explanation.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from flcore.clients.clientimage import ImageClient
from flcore.servers.serverbase import Server
from utils.config_utils import IMAGES, MULTI_ROUND, ConfigError, HParam

MODEL_KEYS = ("model", "pretrained", "trainable", "last_k", "lora_rank", "lora_alpha", "lora_targets",
              "prompt_length", "prompt_deep", "image_size", "augment")

TRAIN_HPARAMS = {
    "optimizer": HParam("str", "sgd", "local optimizer, created anew in every round", choices=("sgd", "adam", "adamw")),
    "momentum": HParam("float", 0.0, "SGD momentum (optimizer sgd only)", low=0.0, high=1.0),
    "weight_decay": HParam("float", 0.0, "weight decay of the local optimizer (decoupled for adamw)", low=0.0),
    "lr_schedule": HParam("str", "constant", "learning rate over the rounds of each task: constant, or cosine "
                                             "from the base rate towards 0 (restarting with every task)",
                          choices=("constant", "cosine")),
}


def model_settings(args) -> dict:
    return {k: getattr(args, k) for k in MODEL_KEYS}


def torch_device(args) -> str:
    wanted = str(getattr(args, "device_used", None) or args.device)
    if wanted.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ImageServer(Server):
    INPUT = IMAGES
    REGIME = MULTI_ROUND
    USES_TORCH = True
    CLIENT_CLASS = ImageClient
    HPARAMS = dict(TRAIN_HPARAMS)

    @classmethod
    def check_hparams(cls, hp: dict, sources: dict) -> None:
        if "momentum" in hp and hp.get("optimizer", "sgd") != "sgd" and sources.get("momentum") != "default":
            raise ConfigError(f"hyperparameter momentum = {hp['momentum']} ({sources.get('momentum')}) applies to "
                              f"optimizer sgd only, but optimizer is {hp['optimizer']}. Remove the setting.")

    # ---- setup
    def init_algorithm(self) -> None:
        if torch is None:
            raise RuntimeError(f"{self.algorithm} trains a network and needs torch: install the environment of "
                               f"README section 1 (env_cuda.yaml)")
        from flcore.models.network import build_network, configure_determinism

        configure_determinism()
        self.device = torch_device(self.args)
        self.bundle = build_network(model_settings(self.args), self.num_classes, self.device,
                                    seed=self.seeds["algorithm"])
        self.store.set_pipeline(self.bundle.pipeline)
        for c in self.clients:
            c.attach(self.bundle, self.device, self.rounds_per_task)
        self.global_state = self.bundle.get_state()
        self.global_version = 0
        info = self.bundle.info
        self.logger.info(f"network {self.args.model} ({info['trainable']}): {info['parameters_trainable']:,} of "
                         f"{info['parameters_total']:,} parameters trained, on {self.device}")

    def extra_run_info(self) -> dict:
        return {"network": self.bundle.info, "device": self.device}

    # ---- messages
    def model_message(self, t: int) -> dict:
        """The global state with the head rows of the classes seen up to task t."""
        from flcore.models.network import HEAD_PARAMS

        rows = torch.as_tensor(self.scenario.seen_classes(t), dtype=torch.int64)
        return {"state": {n: (v.index_select(0, rows) if n in HEAD_PARAMS else v)
                          for n, v in self.global_state.items()}}

    def download_payload(self, t: int, r: int, client):
        return self.model_message(t)

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        """FedAvg: the mean of the uploaded states weighted by training samples (no upload: no change)."""
        from flcore.models.network import HEAD_PARAMS, average_states

        if not uploads:
            return
        mean = average_states([(p["state"], int(p["n"])) for _cid, p in uploads])
        rows = torch.as_tensor(self.scenario.seen_classes(t), dtype=torch.int64)
        state = dict(self.global_state)
        for n, v in mean.items():
            if n in HEAD_PARAMS:
                full = state[n].clone()
                full.index_copy_(0, rows, v)
                state[n] = full
            else:
                state[n] = v
        self.global_state = state
        self.global_version += 1

    def final_broadcast_payload(self, t: int):
        return self.model_message(t)

    # ---- the global model
    def _hold_global(self) -> None:
        tag = ("server", self.global_version)
        if self.bundle.holder != tag:
            self.bundle.load_state(self.global_state, None, holder=tag)
        self.bundle.eval_mode()

    def predict(self, X, seen: np.ndarray) -> np.ndarray:
        self._hold_global()
        seen = np.asarray(seen, dtype=np.int64)
        cls = torch.as_tensor(seen, device=self.device)
        with torch.no_grad():
            logits = self.bundle.net(X.to(self.device, non_blocking=True))
            pick = logits.index_select(1, cls).argmax(dim=1)
        return seen[pick.cpu().numpy()]
