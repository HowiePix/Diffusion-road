import os
import torch
import random
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AvgMeter:
    def __init__(self) -> None:
        self.value = [0]

    def update(self, val):
        self.value.append(val)

        while len(self.value) > 100:
            self.value.pop(0)

    def statistic(self):
        return sum(self.value) / len(self.value)

def training_setup(args):

    os.makedirs("./runs", exist_ok=True)
    runs = os.listdir("./runs")
    runs_idx = list(map(lambda x: int(x.rsplit("_")[-1]), runs))
    print(runs_idx)
    cur_runs_idx = max(runs_idx) + 1
    for i in range(cur_runs_idx):
        if i not in runs_idx:
            cur_runs_idx = i
            break

    curr_run_dir = f"./runs/run_{cur_runs_idx}"
    os.makedirs(curr_run_dir, exist_ok=False)

    import json
    from omegaconf import OmegaConf as OCf
    resovled = OCf.to_container(args, resolve=True)
    with open(os.path.join(curr_run_dir, f"run_{cur_runs_idx}.json"), "w") as f:
        json.dump(resovled, fp=f, ensure_ascii=False, indent=4)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        filename=os.path.join(curr_run_dir,'info.log'),
        level=logging.INFO
    )

    return curr_run_dir, logger