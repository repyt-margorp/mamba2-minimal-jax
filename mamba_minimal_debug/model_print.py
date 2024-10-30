"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        print(f"[ModelArgs] Initialized with d_model={self.d_model}, n_layer={self.n_layer}, vocab_size={self.vocab_size}, "
              f"d_state={self.d_state}, expand={self.expand}, dt_rank={self.dt_rank}, d_conv={self.d_conv}, "
              f"pad_vocab_size_multiple={self.pad_vocab_size_multiple}, conv_bias={self.conv_bias}, bias={self.bias}")
              
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            print(f"[ModelArgs] dt_rank set to auto-calculated value: {self.dt_rank}")
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            old_vocab_size = self.vocab_size
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
            print(f"[ModelArgs] Adjusted vocab_size from {old_vocab_size} to {self.vocab_size} to be a multiple of {self.pad_vocab_size_multiple}")


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        print("[Mamba] Initializing Mamba model.")
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        print(f"  [Mamba] Initialized embedding with vocab_size={args.vocab_size} and d_model={args.d_model}")
        print(f"  [Mamba] embedding.weight shape: {self.embedding.weight.shape}")
        print(f"  [Mamba] embedding.weight sample values: {self.embedding.weight.flatten()[:5]}")
        
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        print(f"  [Mamba] Initialized {args.n_layer} ResidualBlocks.")
        
        self.norm_f = RMSNorm(args.d_model)
        print(f"  [Mamba] Initialized final RMSNorm with d_model={args.d_model}")
        print(f"  [Mamba] norm_f.weight shape: {self.norm_f.weight.shape}")
        print(f"  [Mamba] norm_f.weight sample values: {self.norm_f.weight.flatten()[:5]}")
        
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
        print("  [Mamba] Initialized LM Head and tied weights with embedding.")
        print(f"  [Mamba] lm_head.weight shape: {self.lm_head.weight.shape}")
        print(f"  [Mamba] lm_head.weight sample values: {self.lm_head.weight.flatten()[:5]}")

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        print(f"[Mamba.forward] Input_ids shape: {input_ids.shape}")
        print(f"[Mamba.forward] input_ids sample values: {input_ids.flatten()[:5]}")
        
        x = self.embedding(input_ids)
        print(f"[Mamba.forward] Embedding output shape: {x.shape}")
        print(f"[Mamba.forward] Embedding output sample values: {x.flatten()[:5]}")
        
        for idx, layer in enumerate(self.layers):
            print(f"[Mamba.forward] Processing layer {idx + 1}/{self.args.n_layer}")
            x = layer(x)
            print(f"  [Mamba.forward] After layer {idx + 1} shape: {x.shape}")
            print(f"  [Mamba.forward] After layer {idx + 1} sample values: {x.flatten()[:5]}")
        
        x = self.norm_f(x)
        print(f"[Mamba.forward] After final RMSNorm shape: {x.shape}")
        print(f"[Mamba.forward] After final RMSNorm sample values: {x.flatten()[:5]}")
        
        logits = self.lm_head(x)
        print(f"[Mamba.forward] Logits shape: {logits.shape}")
        print(f"[Mamba.forward] Logits sample values: {logits.flatten()[:5]}")
        
        return logits
    

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            config = json.load(open(resolved_archive_file))
            print(f"[Mamba.from_pretrained] Loaded config from {resolved_archive_file}")
            print(f"  [Mamba.from_pretrained] Config data: {config}")
            return config
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            state_dict = torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
            print(f"[Mamba.from_pretrained] Loaded state_dict from {resolved_archive_file}")
            print(f"  [Mamba.from_pretrained] state_dict keys: {list(state_dict.keys())[:5]}")
            return state_dict
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        print(f"[Mamba.from_pretrained] ModelArgs created: {args}")
        
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
            print(f"  [Mamba.from_pretrained] Mapping {key} to {new_key}")
        model.load_state_dict(new_state_dict)
        print("[Mamba.from_pretrained] State dict loaded into model.")
        
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        print("[ResidualBlock] Initializing ResidualBlock.")
        
        self.mixer = MambaBlock(args)
        print("  [ResidualBlock] Initialized MambaBlock.")
        print(f"    [ResidualBlock] mixer.in_proj.weight shape: {self.mixer.in_proj.weight.shape}")
        print(f"    [ResidualBlock] mixer.in_proj.weight sample values: {self.mixer.in_proj.weight.flatten()[:5]}")
        
        self.norm = RMSNorm(args.d_model)
        print(f"  [ResidualBlock] Initialized RMSNorm with d_model={args.d_model}")
        print(f"    [ResidualBlock] norm.weight shape: {self.norm.weight.shape}")
        print(f"    [ResidualBlock] norm.weight sample values: {self.norm.weight.flatten()[:5]}")
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        print("[ResidualBlock.forward] Input shape:", x.shape)
        print("[ResidualBlock.forward] Input sample values:", x.flatten()[:5])
        
        norm_x = self.norm(x)
        print(f"  [ResidualBlock.forward] After RMSNorm shape: {norm_x.shape}")
        print(f"  [ResidualBlock.forward] After RMSNorm sample values: {norm_x.flatten()[:5]}")
        
        mixer_output = self.mixer(norm_x)
        print(f"  [ResidualBlock.forward] MambaBlock output shape: {mixer_output.shape}")
        print(f"  [ResidualBlock.forward] MambaBlock output sample values: {mixer_output.flatten()[:5]}")
        
        output = mixer_output + x
        print(f"  [ResidualBlock.forward] After residual connection shape: {output.shape}")
        print(f"  [ResidualBlock.forward] After residual connection sample values: {output.flatten()[:5]}")
        
        return output
    

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args
        print("[MambaBlock] Initializing MambaBlock.")
        
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        print(f"  [MambaBlock] Initialized in_proj with input_dim={args.d_model}, output_dim={self.args.d_inner * 2}")
        print(f"  [MambaBlock] in_proj.weight shape: {self.in_proj.weight.shape}")
        print(f"  [MambaBlock] in_proj.weight sample values: {self.in_proj.weight.flatten()[:5]}")
        if self.in_proj.bias is not None:
            print(f"  [MambaBlock] in_proj.bias shape: {self.in_proj.bias.shape}")
            print(f"  [MambaBlock] in_proj.bias sample values: {self.in_proj.bias.flatten()[:5]}")
        
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        print(f"  [MambaBlock] Initialized conv1d with in_channels={args.d_inner}, "
              f"out_channels={args.d_inner}, kernel_size={args.d_conv}, groups={args.d_inner}, padding={args.d_conv - 1}")
        print(f"  [MambaBlock] conv1d.weight shape: {self.conv1d.weight.shape}")
        print(f"  [MambaBlock] conv1d.weight sample values: {self.conv1d.weight.flatten()[:5]}")
        if self.conv1d.bias is not None:
            print(f"  [MambaBlock] conv1d.bias shape: {self.conv1d.bias.shape}")
            print(f"  [MambaBlock] conv1d.bias sample values: {self.conv1d.bias.flatten()[:5]}")
        
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        print(f"  [MambaBlock] Initialized x_proj with input_dim={args.d_inner}, output_dim={self.args.dt_rank + self.args.d_state * 2}")
        print(f"  [MambaBlock] x_proj.weight shape: {self.x_proj.weight.shape}")
        print(f"  [MambaBlock] x_proj.weight sample values: {self.x_proj.weight.flatten()[:5]}")
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        print(f"  [MambaBlock] Initialized dt_proj with input_dim={args.dt_rank}, output_dim={args.d_inner}")
        print(f"  [MambaBlock] dt_proj.weight shape: {self.dt_proj.weight.shape}")
        print(f"  [MambaBlock] dt_proj.weight sample values: {self.dt_proj.weight.flatten()[:5]}")
        if self.dt_proj.bias is not None:
            print(f"  [MambaBlock] dt_proj.bias shape: {self.dt_proj.bias.shape}")
            print(f"  [MambaBlock] dt_proj.bias sample values: {self.dt_proj.bias.flatten()[:5]}")
        
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        print(f"  [MambaBlock] Initialized A_log with shape: {self.A_log.shape}")
        print(f"  [MambaBlock] A_log sample values: {self.A_log.flatten()[:5]}")
        
        self.D = nn.Parameter(torch.ones(args.d_inner))
        print(f"  [MambaBlock] Initialized D with shape: {self.D.shape}")
        print(f"  [MambaBlock] D sample values: {self.D.flatten()[:5]}")
        
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        print(f"  [MambaBlock] Initialized out_proj with input_dim={self.args.d_inner}, output_dim={self.args.d_model}")
        print(f"  [MambaBlock] out_proj.weight shape: {self.out_proj.weight.shape}")
        print(f"  [MambaBlock] out_proj.weight sample values: {self.out_proj.weight.flatten()[:5]}")
        if self.out_proj.bias is not None:
            print(f"  [MambaBlock] out_proj.bias shape: {self.out_proj.bias.shape}")
            print(f"  [MambaBlock] out_proj.bias sample values: {self.out_proj.bias.flatten()[:5]}")
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
                
        """
        print(f"[MambaBlock.forward] Input shape: {x.shape}")
        print(f"[MambaBlock.forward] Input sample values: {x.flatten()[:5]}")
        
        (b, l, d) = x.shape
        print(f"  [MambaBlock.forward] Batch size: {b}, Sequence length: {l}, d_model: {d}")
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        print(f"  [MambaBlock.forward] in_proj output shape: {x_and_res.shape}")
        print(f"  [MambaBlock.forward] in_proj output sample values: {x_and_res.flatten()[:5]}")
        if self.in_proj.bias is not None:
            print(f"  [MambaBlock.forward] in_proj.bias sample values: {self.in_proj.bias.flatten()[:5]}")
        
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        print(f"  [MambaBlock.forward] Split into x shape: {x.shape}, res shape: {res.shape}")
        print(f"  [MambaBlock.forward] x sample values: {x.flatten()[:5]}")
        print(f"  [MambaBlock.forward] res sample values: {res.flatten()[:5]}")
        
        x = rearrange(x, 'b l d_in -> b d_in l')
        print(f"  [MambaBlock.forward] After rearrange x shape: {x.shape}")
        print(f"  [MambaBlock.forward] After rearrange x sample values: {x.flatten()[:5]}")
        
        x = self.conv1d(x)[:, :, :l]
        print(f"  [MambaBlock.forward] After conv1d shape: {x.shape}")
        print(f"  [MambaBlock.forward] After conv1d sample values: {x.flatten()[:5]}")
        
        x = rearrange(x, 'b d_in l -> b l d_in')
        print(f"  [MambaBlock.forward] After rearrange back x shape: {x.shape}")
        print(f"  [MambaBlock.forward] After rearrange back x sample values: {x.flatten()[:5]}")
        
        x = F.silu(x)
        print(f"  [MambaBlock.forward] After SiLU activation shape: {x.shape}")
        print(f"  [MambaBlock.forward] After SiLU activation sample values: {x.flatten()[:5]}")
        
        y = self.ssm(x)
        print(f"  [MambaBlock.forward] After SSM shape: {y.shape}")
        print(f"  [MambaBlock.forward] After SSM sample values: {y.flatten()[:5]}")
        
        y = y * F.silu(res)
        print(f"  [MambaBlock.forward] After multiplying with SiLU(res) shape: {y.shape}")
        print(f"  [MambaBlock.forward] After multiplying with SiLU(res) sample values: {y.flatten()[:5]}")
        
        output = self.out_proj(y)
        print(f"  [MambaBlock.forward] After out_proj shape: {output.shape}")
        print(f"  [MambaBlock.forward] After out_proj sample values: {output.flatten()[:5]}")
        
        return output
    

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
                
        """
        print("[MambaBlock.ssm] Starting SSM computation.")
        (d_in, n) = self.A_log.shape
        print(f"  [MambaBlock.ssm] d_in: {d_in}, n: {n}")
        
        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        print(f"  [MambaBlock.ssm] A shape: {A.shape}")
        print(f"  [MambaBlock.ssm] A sample values: {A.flatten()[:5]}")
        
        D = self.D.float()
        print(f"  [MambaBlock.ssm] D shape: {D.shape}")
        print(f"  [MambaBlock.ssm] D sample values: {D.flatten()[:5]}")
        
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        print(f"  [MambaBlock.ssm] x_proj output shape: {x_dbl.shape}")
        print(f"  [MambaBlock.ssm] x_proj output sample values: {x_dbl.flatten()[:5]}")
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        print(f"  [MambaBlock.ssm] Split into delta shape: {delta.shape}, B shape: {B.shape}, C shape: {C.shape}")
        print(f"  [MambaBlock.ssm] delta sample values: {delta.flatten()[:5]}")
        print(f"  [MambaBlock.ssm] B sample values: {B.flatten()[:5]}")
        print(f"  [MambaBlock.ssm] C sample values: {C.flatten()[:5]}")
        
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        print(f"  [MambaBlock.ssm] delta after softplus shape: {delta.shape}")
        print(f"  [MambaBlock.ssm] delta after softplus sample values: {delta.flatten()[:5]}")
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        print(f"  [MambaBlock.ssm] selective_scan output shape: {y.shape}")
        print(f"  [MambaBlock.ssm] selective_scan output sample values: {y.flatten()[:5]}")
        
        return y
    

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        print("[MambaBlock.selective_scan] Starting selective_scan computation.")
        (b, l, d_in) = u.shape
        n = A.shape[1]
        print(f"  [MambaBlock.selective_scan] b: {b}, l: {l}, d_in: {d_in}, n: {n}")
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        print("[MambaBlock.selective_scan] Discretizing A and B.")
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        print(f"  [MambaBlock.selective_scan] deltaA shape: {deltaA.shape}")
        print(f"  [MambaBlock.selective_scan] deltaA sample values: {deltaA.flatten()[:5]}")
        
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        print(f"  [MambaBlock.selective_scan] deltaB_u shape: {deltaB_u.shape}")
        print(f"  [MambaBlock.selective_scan] deltaB_u sample values: {deltaB_u.flatten()[:5]}")
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        print("[MambaBlock.selective_scan] Initializing state and preparing for scan.")
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        print(f"  [MambaBlock.selective_scan] Initial x shape: {x.shape}")
        print(f"  [MambaBlock.selective_scan] Initial x sample values: {x.flatten()[:5]}")
        
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            print(f"  [MambaBlock.selective_scan] Step {i+1}/{l}:")
            print(f"    [Step {i+1}] x shape: {x.shape}")
            print(f"    [Step {i+1}] x sample values: {x.flatten()[:5]}")
            
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            print(f"    [Step {i+1}] y shape: {y.shape}")
            print(f"    [Step {i+1}] y sample values: {y.flatten()[:5]}")
            
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        print(f"  [MambaBlock.selective_scan] Stacked y shape: {y.shape}")
        print(f"  [MambaBlock.selective_scan] Stacked y sample values: {y.flatten()[:5]}")
        
        y = y + u * D
        print(f"  [MambaBlock.selective_scan] After adding D * u shape: {y.shape}")
        print(f"  [MambaBlock.selective_scan] After adding D * u sample values: {y.flatten()[:5]}")
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        print(f"[RMSNorm] Initialized RMSNorm with d_model={d_model}, eps={eps}")
        print(f"  [RMSNorm] weight shape: {self.weight.shape}")
        print(f"  [RMSNorm] weight sample values: {self.weight.flatten()[:5]}")

    def forward(self, x):
        print(f"[RMSNorm.forward] Input shape: {x.shape}")
        print(f"[RMSNorm.forward] Input sample values: {x.flatten()[:5]}")
        
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        print(f"  [RMSNorm.forward] Output shape: {output.shape}")
        print(f"  [RMSNorm.forward] Output sample values: {output.flatten()[:5]}")
        
        return output
