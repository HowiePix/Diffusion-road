# Diffusion-Road

Toy experiment about image generation

Detail: [Diffusion Road to AIGC](https://zhuanlan.zhihu.com/p/13515967630)

### Datasets Used

* CIFAR10
* 

## Methods

### Flow Matching

Refer to [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

![flow_matching_cifar10](vis_image/flow_matching_cifar10.gif "FM")

### DDPM

## How to Use

```python
python train_by_flow_matching.py --config_file ./configs/flow_matching.yaml
```
