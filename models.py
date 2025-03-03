import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torchvision
from typing import Optional
from collections import OrderedDict
from functools import partial
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from timm.models.layers import DropPath, PatchEmbed
from timm.models.vision_transformer import _load_weights
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        #print('shape of hidden_states: ', hidden_states.shape)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )

    D = block.mixer.d_inner
    N = block.mixer.d_state


    print('D is : ', D)
    print('N is : ', N)

    block.layer_idx = layer_idx
    return block

class DotProductAttention(nn.Module) :
    def __init__(self, dropout) :
        super().__init__()
        self.dropout = nn.Dropout(dropout) ### dropout attention weight
    
    def forward(self, Q, K, V) :
        d = Q.shape[-1]
        scores = torch.bmm(Q, K.transpose(-1, -2)) / d ** 0.5 ### \sqrt{d}: scale the variance of <q, k> to 1
        self.attention_weights = torch.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), V)

class MultiHeadAttention(nn.Module) :
    def __init__(self, Q_d, K_d, V_d, hidden_d, num_heads, dropout, bias=True) :
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)
        self.W_q = nn.Linear(Q_d, hidden_d, bias=bias) ## hidden_d = num_heads * D
        self.W_k = nn.Linear(K_d, hidden_d, bias=bias)
        self.W_v = nn.Linear(V_d, hidden_d, bias=bias)
        self.W_o = nn.Linear(hidden_d, hidden_d, bias=bias)
    
    def forward(self, Q, K, V) :
        Q = rearrange(self.W_q(Q), 'B L (H D) -> (B H) L D', H=self.num_heads)
        K = rearrange(self.W_k(K), 'B L (H D) -> (B H) L D', H=self.num_heads)
        V = rearrange(self.W_v(V), 'B L (H D) -> (B H) L D', H=self.num_heads)
        o = self.attention(Q, K, V)
        o_concat = rearrange(o, '(B H) L D -> B L (H D)', H=self.num_heads)
        return self.W_o(o_concat)

class PositionWiseFFN(nn.Module) :
    def __init__(self, input_d, hidden_d, output_d, dropout) :
        super().__init__()
        self.fc1 = nn.Linear(input_d, hidden_d)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_d, output_d)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x) :
        x = self.dropout1(self.act(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

class EncoderBlock(nn.Module) :
    def __init__(self, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer=partial(nn.LayerNorm, eps=1e-6)) :
        super().__init__()
        self.ln1 = norm_layer(hidden_d)
        self.self_attn = MultiHeadAttention(hidden_d, hidden_d, hidden_d, hidden_d, num_heads, dropout=attention_dropout)
        # self.self_attn = Attention(hidden_d, num_heads, 64)
        # self.self_attn = MultiHeadSelfAttention(hidden_d, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = norm_layer(hidden_d)
        self.ffn = PositionWiseFFN(hidden_d, ffn_d, hidden_d, dropout)

    def forward(self, x) :   ### ViT: Norm Add
        x1 = self.ln1(x)
        x1 = self.self_attn(x1, x1, x1)
        # x1 = self.self_attn(x1)
        x1 = self.dropout1(x1)
        x1 = x1 + x

        x2 = self.ln2(x1)
        x2 = self.ffn(x2)
        return x1 + x2




class Encoder(nn.Module) :
    def __init__(self, num_layers, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer=partial(nn.LayerNorm, eps=1e-6)) :
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        drop_path_rate = 0
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        factory_kwargs = {"device": None, "dtype": None}
        dpr = [x.item() for x in torch.linspace(0, 0.1, num_layers)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        rms_norm = True
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            hidden_d, eps=1e-5, **factory_kwargs
        )
        layers_dict = OrderedDict() ## dict that remembers the insertion order
        self.layers = nn.ModuleList(
            [
                create_block(
                    hidden_d,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    bimamba_type="v2",
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(num_layers)
            ]
        )

        #self.ln = norm_layer(hidden_d)

    def forward(self, x) :
        residual = None
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        inference_params = None
        features = []
        i = 0
        for layer in self.layers:
            x, residual = layer(x, residual, inference_params = inference_params)
            features.append(x)
        x = fused_add_norm_fn(
            self.drop_path(x),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
        return x, features

    def flops(self, input_shape=(3, 224, 224)):
        flops = 0
        #from lib.utils.measure import get_flops
        #flops += get_flops(self.patch_embed, input_shape)

        L = 1024
        #if self.if_cls_token:
        #    L += 1
        for layer in self.layers:
            # 1 in_proj
            flops += layer.mixer.in_proj.in_features * layer.mixer.in_proj.out_features * L
            # 2 MambaInnerFnNoOutProj
            # 2.1 causual conv1d
            flops += 2*(L + layer.mixer.d_conv - 1) * layer.mixer.d_inner * layer.mixer.d_conv
            # 2.2 x_proj
            flops += L * layer.mixer.x_proj_b.in_features * layer.mixer.x_proj_b.out_features
            # 2.3 dt_proj
            flops += L * layer.mixer.dt_proj_b.in_features * layer.mixer.dt_proj_b.out_features
            # 2.4 selective scan
            """
            u: r(B D L)
            delta: r(B D L)
            A: r(D N)
            B: r(B N L)
            C: r(B N L)
            D: r(D)
            z: r(B D L)
            delta_bias: r(D), fp32
            """
            D = layer.mixer.d_inner
            N = layer.mixer.d_state

            print('L is ', L)
            print('D is : ', D)
            print('N is : ', N)
            for i in range(2):
                # flops += 9 * L * D * N + 2 * D * L
                # A
                flops += D * L * N
                # B
                flops += D * L * N * 2
                # C
                flops += (D * N + D * N) * L
                # D
                flops += D * L
                # Z
                flops += D * L
            # merge
            #attn = layer.mixer.attn
            #flops += attn.global_reduce.in_features * attn.global_reduce.out_features
            # flops += attn.local_reduce.in_features * attn.local_reduce.out_features * L
            #flops += attn.channel_select.in_features * attn.channel_select.out_features
            # flops += attn.spatial_select.in_features * attn.spatial_select.out_features * L
            # 2.5 out_proj
            flops += L * layer.mixer.out_proj.in_features * layer.mixer.out_proj.out_features
            # layer norm
            flops += L * layer.mixer.out_proj.out_features

        # head
        #flops += self.embed_dim * 1000
        return flops


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

def posemb_sincos_3d(h, w, d, dim, temperature: int = 10000, dtype = torch.float32):
    z = torch.arange(d).unsqueeze(0).unsqueeze(0).expand(h, w, d)
    y = torch.arange(w).unsqueeze(0).unsqueeze(-1).expand(h, w, d)
    x = torch.arange(h).unsqueeze(-1).unsqueeze(-1).expand(h, w, d)
    omega = torch.arange(dim // 6) / (dim // 6 - 1)
    omega = 1.0 / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)
    return pe.type(dtype)

class PatchEmbeddingConv(nn.Module) :
    def __init__(self, patch_size, hidden_d, image_dim=2) :
        super().__init__()
        self.patch_size = patch_size
        self.image_dim = image_dim
        self.hidden_d = hidden_d
        if image_dim == 3 :
            self.conv = nn.Conv3d(3, hidden_d, kernel_size=patch_size, stride=patch_size)
        elif image_dim == 2:
            self.conv = nn.Conv2d(3, hidden_d, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x) :
        ## [B, C, H, W, D]
        x = self.conv(x)

        ## [B, hidden, nh, nw, nd]
        if self.image_dim == 3 :
            x = rearrange(x, 'B hid nh nw nd -> B (nh nw nd) hid')
        elif self.image_dim == 2 :
            x = rearrange(x, 'B hid nh nw -> B (nh nw) hid')
            
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViM(nn.Module) :
    def __init__(self, image_size:list, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels=3, dropout=0.0, attention_dropout=0.0, learnable_pe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)) :
        super().__init__()

        ## super-params
        self.hidden_d = hidden_d
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patch = 1
        for i in image_size : 
            self.num_patch *= i // patch_size
        self.image_dim = len(image_size)
        
        # self.patch_embedding = PatchEmbeddingConv(patch_size, hidden_d, self.image_dim)
        patch_dim = channels * (patch_size ** self.image_dim)
        if self.image_dim == 2 :
            self.patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_size, p2 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
        elif self.image_dim == 3 :
            self.patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
            # self.patch_embedding = nn.Sequential(
            #     Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
            #     nn.LayerNorm(patch_dim),
            #     nn.Linear(patch_dim, hidden_d),
            #     nn.LayerNorm(hidden_d),
            #     nn.Linear(hidden_d, hidden_d),
            #     nn.LayerNorm(hidden_d),
            # )
            # self.patch_embedding = nn.Sequential(
            #     Rearrange("b c (h p1) (w p2) (d p3) -> (b h w d) c p1 p2 p3", p1 = patch_size, p2 = patch_size, p3 = patch_size),
            #     nn.Conv3d(1, hidden_d, 4, 2),
            #     nn.BatchNorm3d(hidden_d),
            #     nn.ReLU(),
            #     nn.AvgPool3d((patch_size - 4 + 1) // 2),
            #     Rearrange("(b n) d p1 p2 p3 -> b n (d p1 p2 p3)", n = self.num_patch),
            # )

        self.encoder = Encoder(num_layers, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer)
        
        ## learnable class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_d))

        ## learnable PE
        ## torch.tensor.normal_: fill tensor with elements sampled from a normal distribution
        if learnable_pe :
            self.pos_embedding = nn.Parameter(torch.empty(1, self.num_patch, hidden_d).normal_(std=0.02)) # BERT
        else :
            if self.image_dim == 2 :
                self.pos_embedding = posemb_sincos_2d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.hidden_d).cuda()
            else :
                self.pos_embedding = posemb_sincos_3d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.image_size[2] // self.patch_size, self.hidden_d).cuda()

        ## classification heads
        self.heads = nn.Linear(hidden_d, num_category)


    def forward(self, x) :
        ## [B, C, H, W, D]
        x = self.patch_embedding(x)

        ## sin-cos positional embedding
        x += self.pos_embedding

        ## [B, L, D], concat class token
        # batch_size = x.shape[0]
        # batch_class_token = self.class_token.expand(batch_size, -1, -1)
        # x = torch.concat([batch_class_token, x], dim=1)

        ## Go through the Transformer
        x = self.encoder(x)

        ## Extract classification info
        # x = x[:, 0, :]
        x = x.mean(dim = 1)

        ## Classifier
        x = self.heads(x)

        return x


class ViT_register(nn.Module) :
    def __init__(self, image_size:list, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels=3, dropout=0.0, attention_dropout=0.0, learnable_pe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)) :
        super().__init__()

        ## super-params
        self.hidden_d = hidden_d
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patch = 1
        for i in image_size : 
            self.num_patch *= i // patch_size
        self.image_dim = len(image_size)
        
        # self.patch_embedding = PatchEmbeddingConv(patch_size, hidden_d, self.image_dim)
        patch_dim = channels * (patch_size ** self.image_dim)
        if self.image_dim == 2 :
            self.patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_size, p2 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
        elif self.image_dim == 3 :
            self.patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
            # self.patch_embedding = nn.Sequential(
            #     Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
            #     nn.LayerNorm(patch_dim),
            #     nn.Linear(patch_dim, hidden_d),
            #     nn.LayerNorm(hidden_d),
            #     nn.Linear(hidden_d, hidden_d),
            #     nn.LayerNorm(hidden_d),
            # )
            # self.patch_embedding = nn.Sequential(
            #     Rearrange("b c (h p1) (w p2) (d p3) -> (b h w d) c p1 p2 p3", p1 = patch_size, p2 = patch_size, p3 = patch_size),
            #     nn.Conv3d(1, hidden_d, 4, 2),
            #     nn.BatchNorm3d(hidden_d),
            #     nn.ReLU(),
            #     nn.AvgPool3d((patch_size - 4 + 1) // 2),
            #     Rearrange("(b n) d p1 p2 p3 -> b n (d p1 p2 p3)", n = self.num_patch),
            # )
        num_register_tokens = 16
        self.register_tokens = nn.Parameter(
            torch.randn(num_register_tokens, hidden_d)
        )
        self.add_pos_embedding = nn.Parameter(torch.empty(1, 16, hidden_d).normal_(std=0.02))

        self.encoder = Encoder(num_layers, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer)
        
        ## learnable class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_d))
        

        ## learnable PE
        ## torch.tensor.normal_: fill tensor with elements sampled from a normal distribution
        if learnable_pe :
            self.pos_embedding = nn.Parameter(torch.empty(1, self.num_patch, hidden_d).normal_(std=0.02)) # BERT
        else :
            if self.image_dim == 2 :
                self.pos_embedding = posemb_sincos_2d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.hidden_d).cuda()
            else :
                self.pos_embedding = posemb_sincos_3d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.image_size[2] // self.patch_size, self.hidden_d).cuda()

        ## classification heads
        self.heads = nn.Linear(hidden_d, num_category)


    def forward(self, x) :
        ## [B, C, H, W, D]
        x = self.patch_embedding(x)
        r = repeat(
            self.register_tokens, 
            'n d -> b n d', 
            b=x.shape[0]
        )


        ## sin-cos positional embedding
        x += self.pos_embedding
        x, ps = pack([x, r], 'b * d ')

        ## [B, L, D], concat class token
        # batch_size = x.shape[0]
        # batch_class_token = self.class_token.expand(batch_size, -1, -1)
        # x = torch.concat([batch_class_token, x], dim=1)

        ## Go through the Transformer
        x = self.encoder(x)
        x, _ = unpack(x, ps, 'b * d')
        ## Extract classification info
        # x = x[:, 0, :]
        x = x.mean(dim = 1)

        ## Classifier
        x = self.heads(x)

        return x

class ViM_MAE(ViM) :
    def __init__(self, image_size:list, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels=3, dropout=0.0, attention_dropout=0.0, learnable_pe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_num_layers=1, decoder_num_heads=None, decoder_hidden_d=None, decoder_ffn_d=None, decoder_dropout=None, decoder_attention_dropout=None, decoder_norm_layer=None, mask_ratio=0.75) :
        super().__init__(image_size, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels, dropout, attention_dropout, learnable_pe, norm_layer)
        self.num_masked = int(self.num_patch * mask_ratio)

        # default decoder superparam
        if decoder_num_heads == None : decoder_num_heads = num_heads 
        if decoder_hidden_d == None : decoder_hidden_d = hidden_d
        if decoder_ffn_d == None : decoder_ffn_d = ffn_d
        if decoder_dropout == None : decoder_dropout = dropout
        if decoder_attention_dropout == None : decoder_attention_dropout = attention_dropout
        if decoder_norm_layer == None : decoder_norm_layer = norm_layer

        self.enc_to_dec = None
        if decoder_hidden_d != hidden_d :
            self.enc_to_dec = nn.Linear(hidden_d, decoder_hidden_d)

        self.decoder = Encoder(decoder_num_layers, decoder_num_heads, decoder_hidden_d, decoder_ffn_d, decoder_dropout, decoder_attention_dropout, decoder_norm_layer)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_d))
        self.proj_head = nn.Linear(hidden_d, patch_size ** self.image_dim)

    def forward(self, x) :
        ## [B, C, H, W, D]
        image_patch = rearrange(x, "b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = self.patch_size, p2 = self.patch_size, p3 = self.patch_size)
        mask = torch.ones_like(image_patch)
        x = self.patch_embedding(x)

        ## Add positional embedding
        x += self.pos_embedding

        ## [B, L, D]
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        ## Generate masked&unmasked patches
        shuffle_idx = torch.rand(batch_size, seq_length).cuda().argsort(dim=1)
        masked_idx, unmasked_idx = shuffle_idx[:, :self.num_masked], shuffle_idx[:, self.num_masked:]
        batch_idx = torch.arange(batch_size).cuda().unsqueeze(-1)
        x_masked, x_unmasked = x[batch_idx, masked_idx], x[batch_idx, unmasked_idx]
        mask[batch_idx, unmasked_idx] = 0

        ## Go through the ViT encoder
        x_unmasked = self.encoder(x_unmasked)

        ## Concat
        mask_embedding = self.mask_token.expand(batch_size, x_masked.shape[1], -1)
        x_shuffled = torch.concat([mask_embedding, x_unmasked], dim=1)

        ## Redo shuffle
        x = torch.empty_like(x)
        x[batch_idx, shuffle_idx] = x_shuffled

        ## Add positional embedding
        x += self.pos_embedding

        ## Decoder
        if self.enc_to_dec != None : x = self.enc_to_dec(x)
        x = self.decoder(x)

        ## Pixel Predictor
        x = self.proj_head(x)
        

        if self.image_dim == 2 :
            x = rearrange(x, " b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=self.image_size[0] // self.patch_size, p1=self.patch_size, p2=self.patch_size)
        elif self.image_dim == 3 :
            x = rearrange(x, "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)", h=self.image_size[0] // self.patch_size, w=self.image_size[1] // self.patch_size, p1 = self.patch_size, p2 = self.patch_size, p3 = self.patch_size)
            mask = rearrange(mask, "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)", h=self.image_size[0] // self.patch_size, w=self.image_size[1] // self.patch_size, p1 = self.patch_size, p2 = self.patch_size, p3 = self.patch_size)
        return x, mask


class ViM_multi(nn.Module) :
    def __init__(self, image_size:list, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels=3, dropout=0.0, attention_dropout=0.0, learnable_pe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)) :
        super().__init__()

        ## super-params
        self.hidden_d = hidden_d
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patch = 1
        for i in image_size : 
            self.num_patch *= i // patch_size
        self.image_dim = len(image_size)
        
        # self.patch_embedding = PatchEmbeddingConv(patch_size, hidden_d, self.image_dim)
        patch_dim = channels * (patch_size ** self.image_dim)
        if self.image_dim == 2 :
            self.patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_size, p2 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
        elif self.image_dim == 3 :
            self.patch_embedding_mri = nn.Sequential(
                Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
            self.patch_embedding_pet = nn.Sequential(
                Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )

        self.encoder = Encoder(num_layers, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer)
        
        ## learnable class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_d))

        ## learnable PE
        ## torch.tensor.normal_: fill tensor with elements sampled from a normal distribution
        if learnable_pe :
            self.pos_embedding = nn.Parameter(torch.empty(1, self.num_patch*2, hidden_d).normal_(std=0.02)) # BERT
            self.pos_embedding_mri = nn.Parameter(torch.empty(1, self.num_patch, hidden_d).normal_(std=0.02)) # BERT
            self.pos_embedding_pet = nn.Parameter(torch.empty(1, self.num_patch, hidden_d).normal_(std=0.02))
        else :
            if self.image_dim == 2 :
                self.pos_embedding = posemb_sincos_2d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.hidden_d).cuda()
            else :
                self.pos_embedding = posemb_sincos_3d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.image_size[2] // self.patch_size, self.hidden_d).cuda()

        ## classification heads
        self.heads = nn.Linear(hidden_d, num_category)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)


    def forward(self, x_mri, x_pet) :
        ## [B, C, H, W, D]
        x_mri = self.patch_embedding_mri(x_mri)
        x_pet = self.patch_embedding_pet(x_pet)

        x_mri = x_mri + self.pos_embedding_mri
        x_pet = x_pet + self.pos_embedding_pet
        x = torch.cat([x_mri, x_pet], dim=1)

        ## [B, L, D], concat class token
        # batch_size = x.shape[0]
        # batch_class_token = self.class_token.expand(batch_size, -1, -1)
        # x = torch.concat([batch_class_token, x], dim=1)

        ## Go through the Transformer
        x, inter_features = self.encoder(x)

        ## Extract classification info
        # x = x[:, 0, :]
        inter_features_out = []
        for inter_feature in inter_features :
            inter_feature = inter_feature.mean(dim = 1)
            #print('shape of inter_feature: ', )
            inter_features_out.append(self.pool(inter_feature))
        x = x.mean(dim = 1)
        feature = self.pool(x)
        #print('shape of feature: ', feature.shape)

        ## Classifier
        x = self.heads(x)

        return x#, feature, inter_features_out


class ViM_multi_mri(nn.Module) :
    def __init__(self, image_size:list, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels=3, dropout=0.0, attention_dropout=0.0, learnable_pe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)) :
        super().__init__()

        ## super-params
        self.hidden_d = hidden_d
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patch = 1
        for i in image_size : 
            self.num_patch *= i // patch_size
        self.image_dim = len(image_size)
        
        # self.patch_embedding = PatchEmbeddingConv(patch_size, hidden_d, self.image_dim)
        patch_dim = channels * (patch_size ** self.image_dim)
        if self.image_dim == 2 :
            self.patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_size, p2 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
        elif self.image_dim == 3 :
            self.patch_embedding_mri = nn.Sequential(
                Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
            self.patch_embedding_pet = nn.Sequential(
                Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )

        self.encoder = Encoder(num_layers, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer)
        
        ## learnable class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_d))

        ## learnable PE
        ## torch.tensor.normal_: fill tensor with elements sampled from a normal distribution
        if learnable_pe :
            self.pos_embedding = nn.Parameter(torch.empty(1, self.num_patch*2, hidden_d).normal_(std=0.02)) # BERT
            self.pos_embedding_mri = nn.Parameter(torch.empty(1, self.num_patch, hidden_d).normal_(std=0.02)) # BERT
            self.pos_embedding_pet = nn.Parameter(torch.empty(1, self.num_patch, hidden_d).normal_(std=0.02))
        else :
            if self.image_dim == 2 :
                self.pos_embedding = posemb_sincos_2d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.hidden_d).cuda()
            else :
                self.pos_embedding = posemb_sincos_3d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.image_size[2] // self.patch_size, self.hidden_d).cuda()

        ## classification heads
        self.heads = nn.Linear(hidden_d, num_category)


    def forward(self, x_mri, x_pet) :
        ## [B, C, H, W, D]
        x_mri = self.patch_embedding_mri(x_mri)
        #x_pet = self.patch_embedding_pet(x_pet)

        x_mri = x_mri + self.pos_embedding_mri
        #x_pet = x_pet + self.pos_embedding_pet
        #x = torch.cat([x_mri, x_pet], dim=1)
        x = x_mri

        ## [B, L, D], concat class token
        # batch_size = x.shape[0]
        # batch_class_token = self.class_token.expand(batch_size, -1, -1)
        # x = torch.concat([batch_class_token, x], dim=1)

        ## Go through the Transformer
        x = self.encoder(x)

        ## Extract classification info
        # x = x[:, 0, :]
        x = x.mean(dim = 1)

        ## Classifier
        x = self.heads(x)

        return x



class ViM_multi_pet(nn.Module) :
    def __init__(self, image_size:list, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels=3, dropout=0.0, attention_dropout=0.0, learnable_pe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)) :
        super().__init__()

        ## super-params
        self.hidden_d = hidden_d
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patch = 1
        for i in image_size : 
            self.num_patch *= i // patch_size
        self.image_dim = len(image_size)
        
        # self.patch_embedding = PatchEmbeddingConv(patch_size, hidden_d, self.image_dim)
        patch_dim = channels * (patch_size ** self.image_dim)
        if self.image_dim == 2 :
            self.patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_size, p2 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
        elif self.image_dim == 3 :
            self.patch_embedding_mri = nn.Sequential(
                Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )
            self.patch_embedding_pet = nn.Sequential(
                Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = patch_size, p2 = patch_size, p3 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, hidden_d),
                nn.LayerNorm(hidden_d),
            )

        self.encoder = Encoder(num_layers, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer)
        
        ## learnable class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_d))

        ## learnable PE
        ## torch.tensor.normal_: fill tensor with elements sampled from a normal distribution
        if learnable_pe :
            self.pos_embedding = nn.Parameter(torch.empty(1, self.num_patch*2, hidden_d).normal_(std=0.02)) # BERT
            self.pos_embedding_mri = nn.Parameter(torch.empty(1, self.num_patch, hidden_d).normal_(std=0.02)) # BERT
            self.pos_embedding_pet = nn.Parameter(torch.empty(1, self.num_patch, hidden_d).normal_(std=0.02))
        else :
            if self.image_dim == 2 :
                self.pos_embedding = posemb_sincos_2d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.hidden_d).cuda()
            else :
                self.pos_embedding = posemb_sincos_3d(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size, self.image_size[2] // self.patch_size, self.hidden_d).cuda()

        ## classification heads
        self.heads = nn.Linear(hidden_d, num_category)


    def forward(self, x_mri, x_pet) :
        ## [B, C, H, W, D]
        #x_mri = self.patch_embedding_mri(x_mri)
        x_pet = self.patch_embedding_pet(x_pet)

        #x_mri = x_mri + self.pos_embedding_mri
        x_pet = x_pet + self.pos_embedding_pet
        #x = torch.cat([x_mri, x_pet], dim=1)
        x = x_pet

        ## [B, L, D], concat class token
        # batch_size = x.shape[0]
        # batch_class_token = self.class_token.expand(batch_size, -1, -1)
        # x = torch.concat([batch_class_token, x], dim=1)

        ## Go through the Transformer
        x = self.encoder(x)

        ## Extract classification info
        # x = x[:, 0, :]
        x = x.mean(dim = 1)

        ## Classifier
        x = self.heads(x)

        return x

def MLP(dim, projection_size, hidden_size=1024, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

class ViM_MAE_multi(ViM_multi) :
    def __init__(self, image_size:list, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels=3, dropout=0.0, attention_dropout=0.0, learnable_pe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_num_layers=1, decoder_num_heads=None, decoder_hidden_d=None, decoder_ffn_d=None, decoder_dropout=None, decoder_attention_dropout=None, decoder_norm_layer=None, mask_ratio=0.75) :
        super().__init__(image_size, patch_size, num_category, num_layers, num_heads, hidden_d, ffn_d, channels, dropout, attention_dropout, learnable_pe, norm_layer)
        self.num_masked = int(self.num_patch * mask_ratio)
        self.num_unmasked = self.num_patch - self.num_masked

        # default decoder superparam
        if decoder_num_heads == None : decoder_num_heads = num_heads 
        if decoder_hidden_d == None : decoder_hidden_d = hidden_d
        if decoder_ffn_d == None : decoder_ffn_d = ffn_d
        if decoder_dropout == None : decoder_dropout = dropout
        if decoder_attention_dropout == None : decoder_attention_dropout = attention_dropout
        if decoder_norm_layer == None : decoder_norm_layer = norm_layer

        self.enc_to_dec = None
        if decoder_hidden_d != hidden_d :
            self.enc_to_dec = nn.Linear(hidden_d, decoder_hidden_d)
        
        self.cross_attention_mri = CrossAttention(hidden_d)
        self.cross_attention_pet = CrossAttention(hidden_d)
        mlp_ratio = 4.0
        mlp_hidden_dim = int(hidden_d * mlp_ratio)
        self.out_norm = norm_layer(hidden_d)
        self.mlp_mri = Mlp(in_features=hidden_d, hidden_features=mlp_hidden_dim)
        self.mlp_pet = Mlp(in_features=hidden_d, hidden_features=mlp_hidden_dim)

        self.decoder_mri = Encoder(decoder_num_layers, decoder_num_heads, decoder_hidden_d, decoder_ffn_d, decoder_dropout, decoder_attention_dropout, decoder_norm_layer)
        self.decoder_pet = Encoder(decoder_num_layers, decoder_num_heads, decoder_hidden_d, decoder_ffn_d, decoder_dropout, decoder_attention_dropout, decoder_norm_layer)
        self.mask_token_mri = nn.Parameter(torch.zeros(1, 1, hidden_d))
        self.mask_token_pet = nn.Parameter(torch.zeros(1, 1, hidden_d))
        self.proj_head_mri = nn.Linear(hidden_d, patch_size ** self.image_dim)
        self.proj_head_pet = nn.Linear(hidden_d, patch_size ** self.image_dim)
        self.softmax_temperature = 0.07
        self.mri_projection =  MLP(hidden_d, hidden_d, 1024)#nn.Linear(hidden_d, 512)
        self.pet_projection =  MLP(hidden_d, hidden_d, 1024)#nn.Linear(hidden_d, 512)
        self.mri_projection_cl = MLP(hidden_d, hidden_d, 1024)
        self.encoder_target = Encoder(num_layers, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer)
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_target.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.m = 0.999

    def generated_masked_tokens(self, x, batch_size, seq_length, mask):
        shuffle_idx = torch.rand(batch_size, seq_length).cuda().argsort(dim=1)
        masked_idx, unmasked_idx = shuffle_idx[:, :self.num_masked], shuffle_idx[:, self.num_masked:]
        batch_idx = torch.arange(batch_size).cuda().unsqueeze(-1)
        x_masked, x_unmasked = x[batch_idx, masked_idx], x[batch_idx, unmasked_idx]
        mask[batch_idx, unmasked_idx] = 0
        return x_masked, x_unmasked, mask, batch_idx, shuffle_idx
    
    def calculate_contrastive_score(self, features, features_c):
        #bz = features.size(0)
        #labels = torch.arange(bz).type_as(feature_c).long()
        features = F.normalize(features, dim=-1)
        features_c = F.normalize(features_c, dim=-1)
        scores = features @ features_c.T 
        scores /= self.softmax_temperature
        #scores1 = scores.T
        #loss0 = F.cross_entropy(scores, labels)
        #loss1 = F.cross_entropy(scores1, labels)

        return scores

    def inter_contrastive_learning(self, x_mri, x_pet):
        # contrastive learning with CLIP
        mri_features = x_mri.mean(dim=1)
        pet_features = x_pet.mean(dim=1)

        mri_features = self.mri_projection_cl(mri_features)

        mri_features = F.normalize(mri_features, dim=-1)
        pet_features = F.normalize(pet_features, dim=-1)

        # cosine similarity as logits
        logits_per_mri = mri_features @ pet_features.t()
        logits_per_mri /= self.softmax_temperature
        logits_per_pet = logits_per_mri.t()
        return logits_per_mri, logits_per_pet

    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_target.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)   

    def forward(self, x_mri, x_pet) :
        ## [B, C, H, W, D]
        image_patch = rearrange(x_mri, "b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1 = self.patch_size, p2 = self.patch_size, p3 = self.patch_size)
        mask_mri = torch.ones_like(image_patch)
        mask_pet = torch.ones_like(image_patch)
        x_mri = self.patch_embedding_mri(x_mri)
        x_pet = self.patch_embedding_pet(x_pet)
        mask_mri_c = torch.ones_like(image_patch)
        mask_pet_c = torch.ones_like(image_patch)

        x_mri = x_mri + self.pos_embedding_mri
        x_pet = x_pet + self.pos_embedding_pet

        #x = torch.concat([x1, x2], dim=1)
        ## Add positional embedding
        #x + = self.pos_embedding

        ## [B, L, D]
        batch_size = x_mri.shape[0]
        seq_length = x_pet.shape[1]

        ## Generate masked&unmasked patches for MRI
        x_masked_mri, x_unmasked_mri, mask_mri, batch_idx_mri, shuffle_idx_mri = self.generated_masked_tokens(x_mri, batch_size, seq_length, mask_mri)
        ## Generate masked&unmasked patches for PET
        x_masked_pet, x_unmasked_pet, mask_pet, batch_idx_pet, shuffle_idx_pet = self.generated_masked_tokens(x_pet, batch_size, seq_length, mask_pet)
        x_unmasked = torch.concat([x_unmasked_mri, x_unmasked_pet], dim=1)
        ## Go through the ViT encoder
        x_unmasked = self.encoder(x_unmasked)
        x_unmasked_mri = x_unmasked[:, :self.num_unmasked, :]
        x_unmasked_pet = x_unmasked[:, self.num_unmasked:, :]
        ## Generate masked&unmasked patches for MRI

        x_masked_mri_c, x_unmasked_mri_c, mask_mri_c, batch_idx_mri_c, shuffle_idx_mri_c = self.generated_masked_tokens(x_mri, batch_size, seq_length, mask_mri_c)
        ## Generate masked&unmasked patches for PET
        x_masked_pet_c, x_unmasked_pet_c, mask_pet_c, batch_idx_pet_c, shuffle_idx_pet_c = self.generated_masked_tokens(x_pet, batch_size, seq_length, mask_pet_c)
        x_unmasked_c = torch.concat([x_unmasked_mri_c, x_unmasked_pet_c], dim=1)
        ## Go through the ViT encoder
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            x_unmasked_c = self.encoder_target(x_unmasked_c)  # keys: NxC
            x_unmasked_c = x_unmasked_c.detach()

        x_unmasked_mri_c = x_unmasked_c[:, :self.num_unmasked, :]
        x_unmasked_pet_c = x_unmasked_c[:, self.num_unmasked:, :]

        logits_per_mri, logits_per_pet = self.inter_contrastive_learning(x_unmasked_mri, x_unmasked_pet)

        x_unmasked_mri_proj = self.mri_projection(x_unmasked_mri.mean(dim=1))
        x_unmasked_mri_c = x_unmasked_mri_c.mean(dim=1)


        #x_unmasked_mri_proj_c = self.mri_projection_c(x_unmasked_mri_c.mean(dim=1))
        x_unmasked_pet_proj = self.pet_projection(x_unmasked_pet.mean(dim=1))
        x_unmasked_pet_c = x_unmasked_pet_c.mean(dim=1)
        #x_unmasked_pet_proj_c = self.pet_projection_c(x_unmasked_pet_c.mean(dim=1))

        score_mri = self.calculate_contrastive_score(x_unmasked_mri_proj, x_unmasked_mri_c)
        score_pet = self.calculate_contrastive_score(x_unmasked_pet_proj, x_unmasked_pet_c)

        ## Concat
        mask_embedding_mri = self.mask_token_mri.expand(batch_size, x_masked_mri.shape[1], -1)
        x_shuffled_mri = torch.concat([mask_embedding_mri, x_unmasked_mri], dim=1)

        ## Redo shuffle
        x_mri = torch.empty_like(x_mri)
        x_mri[batch_idx_mri, shuffle_idx_mri] = x_shuffled_mri

        ## Concat
        mask_embedding_pet = self.mask_token_pet.expand(batch_size, x_masked_pet.shape[1], -1)
        x_shuffled_pet = torch.concat([mask_embedding_pet, x_unmasked_pet], dim=1)

        ## Redo shuffle
        x_pet = torch.empty_like(x_pet)
        x_pet[batch_idx_pet, shuffle_idx_pet] = x_shuffled_pet

        ## Add positional embedding
        x_mri += self.pos_embedding_mri
        x_pet += self.pos_embedding_pet

        

        x_mri = self.cross_attention_mri(x_mri, x_unmasked)
        x_pet = self.cross_attention_pet(x_pet, x_unmasked)
        
        x_mri_new = x_mri + self.mlp_mri(self.out_norm(x_mri))
        x_pet_new = x_pet + self.mlp_pet(self.out_norm(x_pet))

        ## Decoder
        if self.enc_to_dec != None : x = self.enc_to_dec(x)
        x_mri = self.decoder_mri(x_mri_new)
        x_pet = self.decoder_pet(x_pet_new)

        ## Pixel Predictor
        x_mri = self.proj_head_mri(x_mri)
        x_pet = self.proj_head_pet(x_pet)

        if self.image_dim == 2 :
            x = rearrange(x, " b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=self.image_size[0] // self.patch_size, p1=self.patch_size, p2=self.patch_size)
        elif self.image_dim == 3 :
            x_mri = rearrange(x_mri, "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)", h=self.image_size[0] // self.patch_size, w=self.image_size[1] // self.patch_size, p1 = self.patch_size, p2 = self.patch_size, p3 = self.patch_size)
            x_pet = rearrange(x_pet, "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)", h=self.image_size[0] // self.patch_size, w=self.image_size[1] // self.patch_size, p1 = self.patch_size, p2 = self.patch_size, p3 = self.patch_size)
            mask_mri = rearrange(mask_mri, "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)", h=self.image_size[0] // self.patch_size, w=self.image_size[1] // self.patch_size, p1 = self.patch_size, p2 = self.patch_size, p3 = self.patch_size)
            mask_pet = rearrange(mask_pet, "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)", h=self.image_size[0] // self.patch_size, w=self.image_size[1] // self.patch_size, p1 = self.patch_size, p2 = self.patch_size, p3 = self.patch_size)
            
        return x_mri, x_pet, mask_mri, mask_pet, score_mri, score_pet, logits_per_mri, logits_per_pet


if __name__ == "__main__" :
    print(posemb_sincos_3d(8, 8, 8, 384).cuda())



############## ViT Instances #################



def vim_model(args) :
    name = args.model_type
    data_parallel = args.data_parallel
    if name == 'vim_B_mae' :
        model = ViM_MAE(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=12, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'vim_B' :
        model = ViM(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=12, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'vit_B_regnister' :
        model = ViT_register(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=12, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'vim_L_mae' :
        model = ViM_MAE(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=24, num_heads=16, hidden_d=1024, ffn_d=4096, channels=1, learnable_pe=True).cuda()
    elif name == 'vim_L' :
        model = ViM(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=24, num_heads=16, hidden_d=1024, ffn_d=4096, channels=1, learnable_pe=True).cuda()
    elif name == 'vim_6_2400_3074_mae' :
        model = ViM_MAE(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=6, num_heads=12, hidden_d=2400, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'vim_6_2400_3074' :
        model = ViM(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=6, num_heads=12, hidden_d=2400, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'multi_B_mae':
        model = ViM_MAE_multi(image_size=[args.input_D, args.input_H, args.input_W], patch_size=16, num_category=3, num_layers=12, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'multi_B_mae_small':
        model = ViM_MAE_multi(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=6, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'multi_B_small':
        model = ViM_multi(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=6, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'multi_B':
        model = ViM_multi(image_size=[args.input_D, args.input_H, args.input_W], patch_size=16, num_category=3, num_layers=12, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'multi_B_mri':
        model = ViM_multi_mri(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=12, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    elif name == 'multi_B_pet':
        model = ViM_multi_pet(image_size=[args.input_D, args.input_H, args.input_W], patch_size=8, num_category=3, num_layers=12, num_heads=12, hidden_d=768, ffn_d=3072, channels=1, learnable_pe=True).cuda()
    else:
        exit(f'ViM {name} doesn\'t exist')
    
    if data_parallel :
        model = nn.DataParallel(model)
    return model

if __name__ == '__main__':
    num_layers = 12
    num_heads = 12
    hidden_d = 768
    ffn_d = 3072
    dropout = 0.1
    attention_dropout = 0.1
    norm_layer = nn.LayerNorm

    net = Encoder(num_layers, num_heads, hidden_d, ffn_d, dropout, attention_dropout, norm_layer)
    print('initialized')
    flops = net.flops()

    print('flops: ', flops)


