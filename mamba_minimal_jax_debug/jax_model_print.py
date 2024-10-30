"""Simple, minimal implementation of Mamba in one file of JAX with added debug prints.

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
import torch # For loading pretrained weights
import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal, normal
import flax
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

from typing import Union, Dict

import math


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

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
        print(f"[ModelArgs] Initialized with d_model={self.d_model}, n_layer={self.n_layer}, vocab_size={self.vocab_size}, "
              f"d_state={self.d_state}, expand={self.expand}, dt_rank={self.dt_rank}, d_conv={self.d_conv}, "
              f"pad_vocab_size_multiple={self.pad_vocab_size_multiple}, conv_bias={self.conv_bias}, bias={self.bias}")
        print(f"[ModelArgs] Calculated d_inner={self.d_inner}")


class Mamba(nn.Module):
    args: ModelArgs

    def setup(self):
        """Full Mamba model."""
        super().__init__()
        print("[Mamba] Initializing Mamba model.")

        self.embedding = nn.Embed(self.args.vocab_size, self.args.d_model)
        print(f"  [Mamba] Initialized embedding with vocab_size={self.args.vocab_size} and d_model={self.args.d_model}")

        self.layers = [ResidualBlock(self.args) for _ in range(self.args.n_layer)]
        print(f"  [Mamba] Initialized {self.args.n_layer} ResidualBlocks.")

        self.norm_f = RMSNorm(self.args.d_model)
        print(f"  [Mamba] Initialized final RMSNorm with d_model={self.args.d_model}")

    def attend(self, input):
        """Use for weight sharing to produce output logits of model"""
        return self.embedding.attend(input)

    @nn.compact
    def __call__(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        print(f"[Mamba.__call__] Input_ids shape: {input_ids.shape}")
        print(f"[Mamba.__call__] Input_ids sample values: {input_ids.flatten()[:5]}")

        x = self.embedding(input_ids)
        print(f"[Mamba.__call__] Embedding output shape: {x.shape}")
        print(f"[Mamba.__call__] Embedding output sample values: {x.flatten()[:5]}")

        for idx, layer in enumerate(self.layers):
            print(f"[Mamba.__call__] Processing layer {idx + 1}/{self.args.n_layer}")
            x = layer(x)
            print(f"  [Mamba.__call__] After layer {idx + 1} shape: {x.shape}")
            print(f"  [Mamba.__call__] After layer {idx + 1} sample values: {x.flatten()[:5]}")

        x = self.norm_f(x)
        print(f"[Mamba.__call__] After final RMSNorm shape: {x.shape}")
        print(f"[Mamba.__call__] After final RMSNorm sample values: {x.flatten()[:5]}")

        logits = self.attend(x)
        print(f"[Mamba.__call__] Logits shape: {logits.shape}")
        print(f"[Mamba.__call__] Logits sample values: {logits.flatten()[:5]}")

        return logits


    @staticmethod
    def from_pretrained(pretrained_model_name: str, tokenizer=None):
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
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location=torch.device('cpu'), mmap=True)
        
        print(f"[Mamba.from_pretrained] Loading pretrained model: {pretrained_model_name}")
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', 'params.')
            new_state_dict[new_key] = state_dict[key]
        
        rng = jax.random.PRNGKey(7)
        input_ids = tokenizer("hello what is your name", return_tensors='pt').input_ids
        input_ids = jnp.array(input_ids.numpy())
        random_params = model.init(rng, input_ids)
        random_params_flatten = flax.traverse_util.flatten_dict(random_params, sep=".")

        params = convert_from_pytorch(new_state_dict, random_params_flatten)
        print("[Mamba.from_pretrained] Pretrained weights loaded successfully")
        
        return model, params


class ResidualBlock(nn.Module):
    args:ModelArgs

    def setup(self):
        """Full Mamba model."""
        super().__init__()
        print("[ResidualBlock] Initializing ResidualBlock.")
        self.mixer = MambaBlock(self.args)
        self.norm = RMSNorm(self.args.d_model)

    @nn.compact
    def __call__(self, x):
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
        print("[ResidualBlock.__call__] Input shape:", x.shape)
        print("[ResidualBlock.__call__] Input sample values:", x.flatten()[:5])

        output = self.mixer(self.norm(x)) + x
        print(f"[ResidualBlock.__call__] Output shape: {output.shape}")
        print(f"[ResidualBlock.__call__] Output sample values: {output.flatten()[:5]}")

        return output


class MambaBlock(nn.Module):
    args: ModelArgs

    def setup(self):
        print("[MambaBlock] Initializing MambaBlock.")
        self.in_proj = nn.Dense(features=self.args.d_inner * 2,
                                kernel_init=normal(),
                                use_bias=self.args.bias)
        print(f"  [MambaBlock] Initialized in_proj with input_dim={self.args.d_model}, output_dim={self.args.d_inner * 2}")

        self.conv1d = nn.Conv(features=self.args.d_inner,
                              kernel_size=[self.args.d_conv],
                              feature_group_count=self.args.d_inner,
                              padding=self.args.d_conv - 1,
                              use_bias=self.args.conv_bias,
                              )
        print(f"  [MambaBlock] Initialized conv1d with features={self.args.d_inner}, kernel_size={self.args.d_conv}")

        self.x_proj = nn.Dense(self.args.dt_rank + self.args.d_state * 2, use_bias=False)
        print(f"  [MambaBlock] Initialized x_proj with output_dim={self.args.dt_rank + self.args.d_state * 2}")

        self.dt_proj = nn.Dense(self.args.d_inner, use_bias=True)
        print(f"  [MambaBlock] Initialized dt_proj with output_dim={self.args.d_inner}")

        A = jnp.tile(jnp.arange(1, self.args.d_state + 1), (self.args.d_inner, 1))
        self.A_log = self.param('A_log', lambda rng, shape: jnp.log(A), (self.args.d_inner, self.args.d_state))
        print(f"  [MambaBlock] Initialized A_log with shape: {self.A_log.shape}")
        print(f"  [MambaBlock] A_log sample values: {self.A_log.flatten()[:5]}")

        self.D = self.param('D', nn.initializers.ones, (self.args.d_inner,))
        print(f"  [MambaBlock] Initialized D with shape: {self.D.shape}")
        print(f"  [MambaBlock] D sample values: {self.D.flatten()[:5]}")

        self.out_proj = nn.Dense(self.args.d_model, kernel_init=normal(), use_bias=self.args.bias)
        print(f"  [MambaBlock] Initialized out_proj with output_dim={self.args.d_model}")


    def __call__(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        print(f"[MambaBlock.__call__] Input shape: {x.shape}")
        print(f"[MambaBlock.__call__] Input sample values: {x.flatten()[:5]}")

        (b, l, d) = x.shape
        print(f"  [MambaBlock.__call__] Batch size: {b}, Sequence length: {l}, d_model: {d}")

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        print(f"  [MambaBlock.__call__] in_proj output shape: {x_and_res.shape}")
        print(f"  [MambaBlock.__call__] in_proj output sample values: {x_and_res.flatten()[:5]}")

        (x, res) = jnp.split(x_and_res, indices_or_sections=[self.args.d_inner,], axis=-1)
        print(f"  [MambaBlock.__call__] Split into x shape: {x.shape}, res shape: {res.shape}")
        print(f"  [MambaBlock.__call__] x sample values: {x.flatten()[:5]}")
        print(f"  [MambaBlock.__call__] res sample values: {res.flatten()[:5]}")

        x = self.conv1d(x)[:, :l, :]
        print(f"  [MambaBlock.__call__] After conv1d shape: {x.shape}")
        print(f"  [MambaBlock.__call__] After conv1d sample values: {x.flatten()[:5]}")

        x = jax.nn.silu(x)
        print(f"  [MambaBlock.__call__] After SiLU activation shape: {x.shape}")
        print(f"  [MambaBlock.__call__] After SiLU activation sample values: {x.flatten()[:5]}")

        y = self.ssm(x)
        print(f"  [MambaBlock.__call__] After SSM shape: {y.shape}")
        print(f"  [MambaBlock.__call__] After SSM sample values: {y.flatten()[:5]}")

        y = y * jax.nn.silu(res)
        print(f"  [MambaBlock.__call__] After multiplying with SiLU(res) shape: {y.shape}")
        print(f"  [MambaBlock.__call__] After multiplying with SiLU(res) sample values: {y.flatten()[:5]}")

        output = self.out_proj(y)
        print(f"  [MambaBlock.__call__] After out_proj shape: {output.shape}")
        print(f"  [MambaBlock.__call__] After out_proj sample values: {output.flatten()[:5]}")

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

        A = -jnp.exp(self.A_log)  # shape (d_in, n)
        print(f"  [MambaBlock.ssm] A shape: {A.shape}")
        print(f"  [MambaBlock.ssm] A sample values: {A.flatten()[:5]}")

        D = self.D
        print(f"  [MambaBlock.ssm] D shape: {D.shape}")
        print(f"  [MambaBlock.ssm] D sample values: {D.flatten()[:5]}")

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        print(f"  [MambaBlock.ssm] x_proj output shape: {x_dbl.shape}")
        print(f"  [MambaBlock.ssm] x_proj output sample values: {x_dbl.flatten()[:5]}")

        (delta, B, C) = jnp.split(x_dbl, indices_or_sections=[self.args.dt_rank, self.args.dt_rank+n], axis=-1)
        print(f"  [MambaBlock.ssm] Split into delta shape: {delta.shape}, B shape: {B.shape}, C shape: {C.shape}")
        print(f"  [MambaBlock.ssm] delta sample values: {delta.flatten()[:5]}")
        print(f"  [MambaBlock.ssm] B sample values: {B.flatten()[:5]}")
        print(f"  [MambaBlock.ssm] C sample values: {C.flatten()[:5]}")

        delta = jax.nn.softplus(self.dt_proj(delta))  # (b, l, d_in)
        print(f"  [MambaBlock.ssm] delta after softplus shape: {delta.shape}")
        print(f"  [MambaBlock.ssm] delta after softplus sample values: {delta.flatten()[:5]}")

        y = self.selective_scan(x, delta, A, B, C, D)
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

        print("[MambaBlock.selective_scan] Discretizing A and B.")
        deltaA = jnp.exp(jnp.einsum('b l d, d n -> b l d n', delta, A))
        print(f"  [MambaBlock.selective_scan] deltaA shape: {deltaA.shape}")
        print(f"  [MambaBlock.selective_scan] deltaA sample values: {deltaA.flatten()[:5]}")

        deltaB_u = jnp.einsum('b l d, b l n, b l d -> b l d n', delta, B, u)
        print(f"  [MambaBlock.selective_scan] deltaB_u shape: {deltaB_u.shape}")
        print(f"  [MambaBlock.selective_scan] deltaB_u sample values: {deltaB_u.flatten()[:5]}")

        print("[MambaBlock.selective_scan] Initializing state and preparing for scan.")
        x = jnp.zeros((b, d_in, n))
        print(f"  [MambaBlock.selective_scan] Initial x shape: {x.shape}")
        print(f"  [MambaBlock.selective_scan] Initial x sample values: {x.flatten()[:5]}")

        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            print(f"  [MambaBlock.selective_scan] Step {i+1}/{l}:")
            print(f"    [Step {i+1}] x shape: {x.shape}")
            print(f"    [Step {i+1}] x sample values: {x.flatten()[:5]}")

            y = jnp.einsum('b d n, b n -> b d', x, C[:, i, :])
            print(f"    [Step {i+1}] y shape: {y.shape}")
            print(f"    [Step {i+1}] y sample values: {y.flatten()[:5]}")

            ys.append(y)

        y = jnp.stack(ys, axis=1)  # shape (b, l, d_in)
        print(f"  [MambaBlock.selective_scan] Stacked y shape: {y.shape}")
        print(f"  [MambaBlock.selective_scan] Stacked y sample values: {y.flatten()[:5]}")

        y = y + u * D
        print(f"  [MambaBlock.selective_scan] After adding D * u shape: {y.shape}")
        print(f"  [MambaBlock.selective_scan] After adding D * u sample values: {y.flatten()[:5]}")

        return y


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        print(f"[RMSNorm.__call__] Input shape: {x.shape}")
        print(f"[RMSNorm.__call__] Input sample values: {x.flatten()[:5]}")

        weight = self.param('weight', nn.initializers.ones, (self.d_model,))
        print(f"  [RMSNorm.__call__] weight shape: {weight.shape}")
        print(f"  [RMSNorm.__call__] weight sample values: {weight.flatten()[:5]}")

        normed = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        print(f"  [RMSNorm.__call__] After normalization shape: {normed.shape}")
        print(f"  [RMSNorm.__call__] After normalization sample values: {normed.flatten()[:5]}")

        output = normed * weight
        print(f"  [RMSNorm.__call__] Output shape: {output.shape}")
        print(f"  [RMSNorm.__call__] Output sample values: {output.flatten()[:5]}")

        return output

def convert_from_pytorch(pt_state: Dict, params_flatten):
    """Convert PyTorch state dict to JAX params."""
    print("[convert_from_pytorch] Starting conversion from PyTorch to JAX")
    jax_state = dict(pt_state)

    for key, tensor in pt_state.items():
        tensor = tensor.cpu().numpy()

        if "embedding.weight" in key:
            del jax_state[key]
            key = key.replace("embedding.weight", "embedding.embedding")
            jax_state[key] = tensor
        
        if "layers." in key:
            del jax_state[key]
            key = key.replace("layers.", "layers_")
            jax_state[key] = tensor

        if "proj.weight" in key:
            del jax_state[key]
            key = key.replace("proj.weight", "proj.kernel")
            jax_state[key] = tensor

        if "conv1d.weight" in key:
            del jax_state[key]
            key = key.replace("conv1d.weight", "conv1d.kernel")
            jax_state[key] = tensor
        
        if "lm_head" in key:
            del jax_state[key]

    jax_state_transposed = {}

    for key in params_flatten.keys():
        if params_flatten[key].shape != jax_state[key].shape:
            jax_state_transposed[key] = jax_state[key].T
        else:
            jax_state_transposed[key] = jax_state[key]

        if params_flatten[key].dtype != jax_state[key].dtype:
            jax_state_transposed[key] = jax_state_transposed[key].numpy()
        else:
            jax_state_transposed[key] = jax_state_transposed[key]

        assert params_flatten[key].shape == jax_state_transposed[key].shape, f'The shape of {key} is not the same with param shape {params_flatten[key].shape} and jax_state shape {jax_state_transposed[key].shape}'
        assert params_flatten[key].dtype == jax_state_transposed[key].dtype, f'The dtype of {key} is not the same with param dtype {params_flatten[key].dtype} and jax_state dtype {jax_state_transposed[key].dtype}'

    params = flax.traverse_util.unflatten_dict(jax_state_transposed, sep=".")
    print("[convert_from_pytorch] Conversion completed successfully")

    return params


if __name__ == '__main__':
    # Test for RMSNorm
    print("[Main] Starting RMSNorm test")

    # Generate a random example input
    rng = jax.random.PRNGKey(0)
    input_shape = (10, 20)  # example shape
    x = jax.random.normal(rng, input_shape)
    print(f"[Main] Generated random input with shape: {x.shape}")
    print(f"[Main] Input sample values: {x.flatten()[:5]}")

    # Initialize the model
    d_model = 20  # should match the last dimension of the input
    rms_norm = RMSNorm(d_model=d_model)
    print(f"[Main] Initialized RMSNorm with d_model={d_model}")

    # Initialize parameters
    params = rms_norm.init(rng, x)
    print("[Main] Initialized RMSNorm parameters")

    # Apply the model
    output = rms_norm.apply(params, x)
    print(f"[Main] RMSNorm output shape: {output.shape}")
    print(f"[Main] RMSNorm output sample values: {output.flatten()[:5]}")

    print("[Main] RMSNorm test completed")
