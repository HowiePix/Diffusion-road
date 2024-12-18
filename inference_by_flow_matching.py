import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image
import argparse
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader

import logging

from tqdm import tqdm
from omegaconf import OmegaConf as of

import models
import methods
from common.registry import registry
from common.utils import seed_everything, AvgMeter, training_setup

# from methods import generate_samples

def load_model(cfg):
    print(registry.class_name_dict)
    model_cls_name = registry.get_model_class(cfg.arch)
    return model_cls_name.from_config(cfg.config)

def load_method(cfg):
    print(registry.class_name_dict)
    method_cls_name = registry.get_method_class(cfg.arch)
    return method_cls_name.from_config(cfg.config)


def infer(model, step, save_dir, device, global_step=None, classifier_guidance=7.5):
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    model_ = copy.deepcopy(model)

    dt = 1.0 / step

    x_t = torch.randn(36, 3, 32, 32).to(device)

    with torch.no_grad():
        for j in range(step):
            t = torch.tensor([j * dt], device=device)

            pred = model_(x_t, t, torch.tensor([1], device=device))
            if classifier_guidance >= 1:
                pred_free = model_(x_t, t, torch.tensor([-1], device=device))
                pred = pred_free + classifier_guidance * (pred - pred_free)

            x_t = x_t + pred * dt

            inter_t = x_t.view([-1, 3, 32, 32]).clip(-1, 1)
            inter_t = inter_t/2 + 0.5

            save_image(inter_t, os.path.join(save_dir, f"generated_FM_images_step_{j}.png"), nrow=6)

        x_t = x_t.view([-1, 3, 32, 32]).clip(-1, 1)
        x_t = x_t/2 + 0.5


        save_image(x_t, os.path.join(save_dir, f"generated_FM_images_step_{step}.png"), nrow=6)


    model.train()

def main(args):
    # curr_log_dir, logger = training_setup(args)

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    model_args, method_args, train_args, data_args = args.model, args.method, args.train, args.data
    model = load_model(model_args)
    method = load_method(method_args)

    ckpt = torch.load(model_args.pretrained_ckpt)["model"]
    model.load_state_dict(ckpt)
    model = model.to(device)

    infer(model, step=50, save_dir="./infer", device=device)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-file", type=str, default="./configs/flow_matching.yaml")

    args = parser.parse_args()
    seed_everything(args.seed)

    args = of.load(args.cfg_file)

    print(args)

    main(args)