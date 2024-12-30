import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import copy
import random as rd
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
    model_cls_name = registry.get_model_class(cfg.arch)
    return model_cls_name.from_config(cfg.config)

def load_method(cfg):
    method_cls_name = registry.get_method_class(cfg.arch)
    return method_cls_name.from_config(cfg.config)


def infer(model, step, save_dir, device, global_step=None, classifier_guidance=3.5):
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
                pred = pred + classifier_guidance * (pred - pred_free)
            x_t = x_t + pred * dt

        x_t = x_t.view([-1, 3, 32, 32]).clip(-1, 1)
        x_t = x_t/2 + 0.5

    if global_step is None:
        save_image(x_t, os.path.join(save_dir, f"generated_FM_images_step_{step}.png"), nrow=6)
    else:
        save_image(x_t, os.path.join(save_dir, f"generated_FM_images_step_{step}_{global_step}.png"), nrow=6)

    model.train()

def main(args):
    curr_log_dir, logger = training_setup(args)

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    model_args, method_args, train_args, data_args = args.model, args.method, args.train, args.data
    model = load_model(model_args).to(device)
    method = load_method(method_args)

    cifar10 = CIFAR10(
        root="./data",
        train=True,
        transform=T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        download=True
    )
    loader = DataLoader(
        dataset=cifar10,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        shuffle=True
    )

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=train_args.learning_rate
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=500,
        gamma=0.8
    )

    global_step = 0

    all_steps = len(loader) * train_args.max_epoches
    if train_args.max_train_steps is not None:
        all_steps = min(all_steps, train_args.max_train_steps)

    logger.info(f"Train {all_steps} steps!")

    progress_bar = tqdm(
        range(0, all_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        # disable=not accelerator.is_local_main_process,
    )

    loss_meter = AvgMeter()

    use_cfg = train_args.classifier_free
    cond_drop_rate = train_args.cond_drop_rate

    for epoch in range(1, train_args.max_epoches+1):
        for step, batch in enumerate(loader):
            x1, y = batch
            x1 = x1.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # bsz = x1.shape[0]
            x0 = torch.randn_like(x1)

            t, xt, ut = method.sample(x0, x1)

            if use_cfg and rd.random() < cond_drop_rate:
                y = -torch.ones_like(y, device=y.device)
                # xt = torch.cat([xt, xt.clone()], dim=0)
                # t = torch.cat([t, t.clone()], dim=0)
                # y = torch.cat([y, -torch.ones_like(y, device=y.device)], dim=0)
                # x1 = torch.cat([x1, x1.clone()], dim=0)
                # x0 = torch.cat([x0, x0.clone()], dim=0)
                # ut = torch.cat([ut, ut.clone()], dim=0)

            pred = model(xt, t, cond=y)

            loss = F.mse_loss(pred, ut)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
            loss_meter.update(loss.detach().item())

            optimizer.step()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": loss_meter.statistic(),
                "lr": scheduler.get_last_lr()[0]
            })

            if global_step % train_args.validation_step == 0:

                # ori_img = T.ToPILImage()(x1[0])
                # ori_img.save(f"./images/test_original_img_{global_step}.png")
                # pred_img = T.ToPILImage()((pred+x0)[0])
                # pred_img.save(f"./images/test_pred_img_{global_step}.png")
                if global_step % (train_args.validation_step*10) == 0:
                    infer(model, step=100, save_dir=os.path.join(curr_log_dir, "images"), device=device, global_step=global_step)
                else:
                    infer(model, step=100, save_dir=os.path.join(curr_log_dir, "images"), device=device)

            if global_step % train_args.checkpoint_step == 0:
                torch.save({
                    "model": model.state_dict()
                }, os.path.join(curr_log_dir, f"checkpoint-{global_step}.pth.tar.gz"))

            if global_step >= all_steps:
                break
        scheduler.step()

    torch.save({
        "model": model.state_dict()
    }, os.path.join(curr_log_dir, "checkpoint-final.pth.tar.gz"))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-file", type=str, default="./configs/flow_matching.yaml")

    args = parser.parse_args()
    seed_everything(args.seed)

    args = of.load(args.cfg_file)

    print(args)

    main(args)