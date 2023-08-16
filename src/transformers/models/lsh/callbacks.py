import math

import pytorch_lightning as pl
from lightning import Callback

from .hashing.node import LSHLinear
from time import perf_counter


def scheduler(lamb, n_0, t):
    return sum([n_0 * (math.e ** (lamb * i__)) for i__ in range(t - 1)])


class ReHashingCallback(Callback):
    def __init__(self, max_steps: int, initial_schedule: int = 100, lamda: float = 10**-1.5, dilution_step: int = 4):
        super().__init__()
        self.state = {"step": 0, "rehash_scheduler": [0], "next_rehash": 0}

        i = 2
        while self.state["rehash_scheduler"][-1] < max_steps:
            self.state["rehash_scheduler"].append(int(scheduler(lamda, initial_schedule, i)))
            i += 1
        self.state["rehash_scheduler"] = self.state["rehash_scheduler"][::dilution_step] + [
            max_steps,
        ]
        print(f"Max number of steps: {max_steps}")
        print("Rehash Intervals: ", self.state["rehash_scheduler"])

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.state["step"] += 1

        if self.state["step"] >= self.state["rehash_scheduler"][self.state["next_rehash"]]:
            t1 = perf_counter()
            for module in pl_module.modules():
                if isinstance(module, LSHLinear):
                    module.rehash()
            self.state["next_rehash"] += 1
            print(f"Rehashed parameters! {perf_counter()-t1}s")


class AttachCallback(Callback):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for cb in trainer.callbacks:
            if isinstance(cb, ReHashingCallback):
                return
        print(">>>>>>>>>>>>>>>>> ATTACHED CALLBACK <<<<<<<<<<<<<<<<<<<<<<<")
        trainer.callbacks.insert(0, ReHashingCallback(pl_module.get_cnt_training_steps()))
        print("===========================================================")
