import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .utils import timestep_embedding, get_2d_sincos_pos_embed
from common.registry import registry

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        hidden_dim=None,
        activation=nn.GELU,
        norm_layer=None,
        drop_rate=[0, 0],
        bias=True,
    ):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        hid_dim = hidden_dim if hidden_dim is not None else out_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            activation(),
            nn.Dropout(drop_rate[0])
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        self.layer2 = nn.Sequential(
            nn.Linear(hid_dim, out_dim, bias=bias),
            nn.Dropout(drop_rate[1])
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.norm_layer(x)
        x = self.layer2(x)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=False
    ):
        super().__init__()

        assert (embed_dim % num_heads == 0)

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=bias)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=bias)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=bias)

        self.ffn = Mlp(in_dim=embed_dim, out_dim=embed_dim, hidden_dim=embed_dim*2)

        self.scale = 1 / self.head_dim**(0.5)

    def forward(self, x):
        x = rearrange(x, "b n (h d) -> b (n h) d", h=self.num_heads)

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        attn = q @ k.transpose(-1, -2) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = rearrange(out, "b (n h) d -> b n (h d)", h=self.num_heads)
        out = self.ffn(out)

        return attn, out

# class AdaNorm(nn.Module):
#     def __init__(
#         self
#     ):
#         super().__init__()

#     def forward(self, cond_embed):
#         pass

class ImageEmbedder(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=128,
        norm_layer=None,
        reshape=True
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        self.reshape = reshape

    def forward(self, img):

        patches = self.proj(img)

        if self.reshape:
            patches = rearrange(patches, "b c h w -> b (h w) c")
        patches = self.norm_layer(patches)

        return patches

class DiTBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=1.2,
        attn_bias=False,
        ada_bias=True
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=attn_bias
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(in_dim=embed_dim, hidden_dim=int(embed_dim*mlp_ratio))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6*embed_dim, bias=ada_bias)
        )

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        _, attn_x = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_msa.unsqueeze(1) * attn_x
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

@registry.register_model("dit")
class DiT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=8,
        embed_dim=128,
        num_heads=4,
        in_channels=3,
        depth=20,
        mlp_ratio=4.0,
        out_channels=None,
        learning_sigma=False,
        time_embed_dim=None,
        time_embed_type="sinusoidal",
        cond_embed_type="sinusoidal",
        adaln_zero=True,
        num_classes=10,
    ):
        super().__init__()

        self.out_channels = out_channels if out_channels is not None else in_channels
        self.out_channels = out_channels * 2 if learning_sigma else out_channels

        self.im_embed = ImageEmbedder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=in_channels
        )
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)

        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else embed_dim
        self.time_embed_type = time_embed_type
        self.cond_embed_type = cond_embed_type

        self.final_layer = FinalLayer(
            embed_dim=embed_dim,
            patch_size=patch_size,
            out_channels=self.out_channels
        )

        self.pos_embed = nn.Parameter(torch.randn((1, (image_size//patch_size)**2, embed_dim)), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size=image_size//patch_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if adaln_zero:
            self._init_weight()

    # stolen from https://github.com/facebookresearch/DiT/blob/main/models.py
    def _init_weight(self):
        w = self.im_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.im_embed.proj.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, time_step, cond):
        x = self.im_embed(x) + self.pos_embed

        t_emb = self.time_embed(self.get_time_embed(time_step))

        if not torch.all(cond == -1):
            cond_emb = self.get_cond_embed(cond)
            t_emb = t_emb + cond_emb

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.norm_layer(x)
        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x)

        return x

    def unpatchify(self, x):
        """
        x: b n p*p*c
        img: b c h w
        """
        p = self.im_embed.patch_size
        n = x.size(1)
        h = w = int(n**(0.5))
        x = rearrange(
            x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h, w=w, p1=p, p2=p, c=self.out_channels
        )
        return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, T, patch_size**2 * C)
    #     imgs: (N, H, W, C)
    #     """
    #     c = self.out_channels
    #     p = self.im_embed.patch_size
    #     h = w = int(x.shape[1] ** 0.5)
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    #     return imgs


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
        image_size = cfg.get("image_size", 32)
        patch_size = cfg.get("patch_size", 8)
        embed_dim = cfg.get("embed_dim", 128)
        num_heads = cfg.get("num_heads", 4)
        in_channels = cfg.get("in_channels", 3)
        depth = cfg.get("depth", 20)
        mlp_ratio = cfg.get("mlp_ratio", 4.0)
        out_channels = cfg.get("out_channels", 3)
        learning_sigma = cfg.get("learning_sigma", False)
        time_embed_dim = cfg.get("time_embed_dim", None)
        time_embed_type = cfg.get("time_embed_type", "sinusoidal")
        cond_embed_type = cfg.get("cond_embed_type", "sinusoidal")
        num_classes = cfg.get("num_classes", 10)
        adaln_zero = cfg.get("adaln_zero", True)

        return cls(
            image_size, patch_size, embed_dim, num_heads,
            in_channels, depth, mlp_ratio, out_channels, learning_sigma,
            time_embed_dim, time_embed_type, cond_embed_type, adaln_zero,
            num_classes
        )


if __name__=="__main__":
    model = ImageEmbedder()

    x = torch.randn((8, 3, 32, 32))

    out = model(x)

    print(out.shape)