import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3

from einops import rearrange, repeat

from tqdm import tqdm

from common.registry import registry


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

@registry.register_evaluator("fid")
class FID:
    def __init__(
        self,
        device,
        stats_dir,
        sampler,
        n_samples=10000,
    ):
        self.sampler = sampler
        self.device = device
        self.stats_dir = stats_dir
        self.n_samples = n_samples
        self.inception = inception_v3(pretrained=True)
        self.inception.fc = nn.Identity()
        self.inception = self.inception.to(device).eval()

    def cal_feat(self, img):
        if img.size(0) == 1:
            img = repeat(img, "b 1 ... -> b c ...", c=3)

        feat = self.inception(img)

        return feat

    # Stolen from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/fid_evaluation.py
    def cal_or_load_dataset(self, data_loader=None):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            assert (data_loader is not None)
            num_batches = int(math.ceil(self.n_samples / data_loader.batch_size))
            stacked_real_features = []
            print(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(data_loader)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    def fid_score(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        self.sampler.eval()
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        for batch in tqdm(batches):
            fake_samples = self.sampler.sample(batch_size=batch)
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)


if __name__=="__main__":
    fid = FID("cpu")

    feat = fid.cal_feat(torch.randn(4, 3, 224, 224))
