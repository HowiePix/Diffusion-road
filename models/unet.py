import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import timestep_embedding

from common.registry import registry


# 定义卷积块
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation=nn.SiLU()
    ):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        return self.conv(x)

class DownLayer(nn.Module):
    def __init__(self, in_dim, out_dim, time_embed_dim, activation=nn.SiLU()) -> None:
        super().__init__()

        self.time_embed_proj = nn.Linear(time_embed_dim, in_dim)

        self.convlayer = DoubleConv(in_channels=in_dim, out_channels=out_dim, activation=activation)

        if in_dim != out_dim:
            self.residual = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.residual = None

    def forward(self, x, t_emb):

        t_emb = self.time_embed_proj(t_emb)

        res = x
        x = x + t_emb[:, :, None, None]
        # x += t_emb[:, :, None, None]
        x = self.convlayer(x)

        if self.residual is not None:
            res = self.residual(res)

        x = x + res

        return x

class MiddleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, time_embed_dim, activation=nn.SiLU()) -> None:
        super().__init__()

        self.time_embed_proj = nn.Linear(time_embed_dim, in_dim)

        self.convlayer = DoubleConv(in_channels=in_dim, out_channels=out_dim, activation=activation)

        if in_dim != out_dim:
            self.residual = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.residual = None

    def forward(self, x, t_emb):

        t_emb = self.time_embed_proj(t_emb)

        res = x
        x = x + t_emb[:, :, None, None]
        x = self.convlayer(x)

        if self.residual is not None:
            res = self.residual(res)

        x = x + res

        return x

class UpLayer(nn.Module):
    def __init__(self, in_dim, out_dim, time_embed_dim, activation=nn.SiLU(), unsample=True) -> None:
        super().__init__()

        self.time_embed_proj = nn.Linear(time_embed_dim, in_dim+out_dim)

        self.convlayer = DoubleConv(in_channels=in_dim+out_dim, out_channels=out_dim, activation=activation)

        self.residual = nn.Conv2d(in_dim+out_dim, out_dim, kernel_size=1)
        if unsample:
            self.unsample = nn.Upsample(scale_factor=2)
        else:
            self.unsample = nn.Identity()

    def forward(self, dx, ex, t_emb):

        t_emb = self.time_embed_proj(t_emb)

        x = torch.cat((self.unsample(dx), ex), dim=1)

        res = x
        x = x + t_emb[:, :, None, None]
        # x += t_emb[:, :, None, None]
        x = self.convlayer(x)

        res = self.residual(res)

        x = x + res

        return x


# 定义UNet模型
@registry.register_model("unet")
class UNet(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        model_channels,
        num_resblock=2,
        time_embed_dim=64,
        time_embed_type="sinusoidal",
        cond_embed_type="sinusoidal",
        num_classes=10,
    ):
        super(UNet, self).__init__()

        # time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # self.cond_embed = nn.Embedding(num_classes, time_embed_dim)
        self.time_embed_dim = time_embed_dim
        self.time_embed_type = time_embed_type
        self.cond_embed_type = cond_embed_type

        self.encoder1 = DownLayer(in_channels, model_channels, time_embed_dim)
        self.encoder2 = DownLayer(model_channels, model_channels*2, time_embed_dim)
        self.encoder3 = DownLayer(model_channels*2, model_channels*4, time_embed_dim)
        self.encoder4 = DownLayer(model_channels*4, model_channels*8, time_embed_dim)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MiddleLayer(model_channels*8, model_channels*8, time_embed_dim)

        self.decoder4 = UpLayer(model_channels*8, model_channels*8, time_embed_dim)
        self.decoder3 = UpLayer(model_channels*8, model_channels*4, time_embed_dim)
        self.decoder2 = UpLayer(model_channels*4, model_channels*2, time_embed_dim)
        self.decoder1 = UpLayer(model_channels*2, model_channels, time_embed_dim)

        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=1)

    def forward(self, x, time_step, cond):

        t_emb = self.time_embed(self.get_time_embed(time_step))

        if not torch.all(cond == -1):
            cond_emb = self.get_cond_embed(cond)
            t_emb = t_emb + cond_emb

        # 编码器路径
        e1 = self.encoder1(x, t_emb)
        e2 = self.encoder2(self.pool(e1), t_emb)
        e3 = self.encoder3(self.pool(e2), t_emb)
        e4 = self.encoder4(self.pool(e3), t_emb)

        # 瓶颈层
        b = self.bottleneck(self.pool(e4), t_emb)

        # 解码器路径
        d4 = self.decoder4(b, e4, t_emb)
        d3 = self.decoder3(d4, e3, t_emb)
        d2 = self.decoder2(d3, e2, t_emb)
        d1 = self.decoder1(d2, e1, t_emb)

        # 输出层
        out = self.final_conv(d1)
        return out

    def get_time_embed(self, t):

        if self.time_embed_type == "sinusoidal":
            return timestep_embedding(t, dim=self.time_embed_dim)
        elif self.time_embed_type == "constant":
            t_emb = t[:, None].repeat(1, self.time_embed_dim)
            return t_emb

    def get_cond_embed(self, cond):

        if cond is None:
            return cond[:, None].repeat(1, self.time_embed_dim)

        if self.cond_embed_type == "sinusoidal":
            return timestep_embedding(cond, dim=self.time_embed_dim)
        elif self.cond_embed_type == "constant":
            cond_emb = cond[:, None].repeat(1, self.time_embed_dim)
            return cond_emb

    @classmethod
    def from_config(cls, cfg):
        in_channels = cfg.get("in_channels", 3)
        out_channels = cfg.get("out_channels", 3)
        model_channels = cfg.get("model_channels", 128)
        num_res_blocks = cfg.get("num_res_blocks", 2)
        time_embed_dim = cfg.get("time_embed_dim", 64)
        time_embed_type = cfg.get("time_embed_type", "sinusoidal")
        cond_embed_type = cfg.get("cond_embed_type", "sinusoidal")
        num_classes = cfg.get("num_classes", 10)

        return cls(
            in_channels,
            out_channels,
            model_channels,
            num_res_blocks,
            time_embed_dim,
            time_embed_type,
            cond_embed_type,
            num_classes
        )

    def from_pretrained(self, ckpt):
        model_ckpt = torch.load(ckpt)["model"]
        self.load_state_dict(model_ckpt)


# 检查模型是否正常运行
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 32, 32)  # 输入为 (batch_size, in_channels, height, width)
    y = model(x)
    print(y.shape)  # 输出维度应为 (batch_size, out_channels, height, width)
