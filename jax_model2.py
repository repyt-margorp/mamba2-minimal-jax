"""
mamba2-minimal-jax
==============

A minimal, single-file implementation of the Mamba-2 model in JAX/Flax.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

import json
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn_flax
from flax.core import freeze, unfreeze
import flax

def torch_to_jax(x, dtype=None):
    if dtype is None:
        return jnp.array(x.detach().cpu().numpy())
    else:
        return jnp.array(x.detach().cpu().numpy(), dtype=dtype)
def jax_to_torch(x):
    #return torch.tensor(np.array(x), dtype=torch.float32)
    return torch.tensor(np.array(x))

def split_tensor(x, split_sizes, dim, use_jax=True):
    """
    Splits the tensor x into multiple tensors based on split_sizes along dimension dim.
    Uses JAX for splitting if use_jax=True, otherwise uses PyTorch.

    Args:
        x (Tensor or jnp.ndarray): The input tensor to split.
        split_sizes (list of int): The sizes to split the tensor into.
        dim (int): The dimension along which to split.
        use_jax (bool): Whether to use JAX for splitting.

    Returns:
        list of Tensor or jnp.ndarray: The list of split tensors.
    """
    if use_jax:
        splits = []
        start = 0
        for size in split_sizes:
            end = start + size
            # generate JAX indices
            indices = jnp.arange(start, end)
            # use JAX's take to split
            split = jnp.take(x, indices=indices, axis=dim)
            splits.append(split)
            start = end
        return splits
    else:
        splits_torch = torch.split(
                jax_to_torch(x), split_sizes, dim=dim
        )
        splits_jax = []
        for spl in splits_torch:
            splits_jax.append(torch_to_jax(spl))
        return tuple(splits_jax)


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


@dataclass
class InferenceCache:
    conv_state: jnp.ndarray  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: jnp.ndarray   # (batch, nheads, headdim, d_state)

    @staticmethod
    #def alloc(batch_size: int, args: 'Mamba2Config') -> 'InferenceCache':
    def alloc(batch_size: int, args: Mamba2Config) -> 'InferenceCache':

        conv_state = jnp.zeros(
            (batch_size, args.d_inner + 2 * args.d_state, args.d_conv)
        )

        ssm_state = jnp.zeros(
            (batch_size, args.nheads, args.headdim, args.d_state)
        )

        return InferenceCache(conv_state, ssm_state)

def print_state_dict_shapes(state_dict):
    """
    A print function that shows the layer name and tensor shape for PyTorch's state_dict.
    
    Args:
    - state_dict (OrderedDict): PyTorch's state_dict。
    
    Returns:
    - None: Print each layer name and the tensor shape to stdout.
    """
    print("\n==== PyTorch State Dict ====\n")
    for layer_name, tensor in state_dict.items():
        print(f"{layer_name}: {tensor.shape}")


class Mamba2LMHeadModel(nn_flax.Module):
    args: Mamba2Config
    use_jax: bool = True

    def setup(self):
        args = self.args

        self.embed = nn_flax.Embed(num_embeddings=args.vocab_size, features=args.d_model)
        self.norms = [RMSNorm(args.d_model) for _ in range(args.n_layer)]
        self.layers = [Mamba2(args, True) for _ in range(args.n_layer)]
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn_flax.Dense(features=args.vocab_size, use_bias=False)

    def __call__(self, input_ids, step_mode, h):
        """
        Arguments
            input_ids: (batch, seqlen) tokens from EleutherAI/gpt-neox-20b tokenizer
            h: hidden states for inference step. If present the constant-time
               (wrt sequence length) inference path will be taken, input_ids
               should have shape (batch, 1) containing the next batch of prompt
               token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing input_ids
        """
        seqlen = input_ids.shape[1]

        x = self.embed(input_ids)

        next_h = [None for _ in range(self.args.n_layer)]

        for i in range(self.args.n_layer):
            flax_norm = self.norms[i]
            flax_layer = self.layers[i]

            normed_x = flax_norm(x)
            #y, h[i] = flax_layer( # this fails
            y, next_h[i] = flax_layer(
                        normed_x,
                        step_mode,
                        h[i])
            x = y + x

        x = self.norm_f(x)

        if self.use_jax:
            logits = self.lm_head(x)
            #logits = jax_to_torch(logits)
        else:
            x_torch = jax_to_torch(x)
            logits = self.torch_lm_head(x_torch)
            logits = torch_to_jax(logits)

        return logits[:, :seqlen], cast(list[InferenceCache], next_h)
        #return x, h, seqlen, logits # this fails
        #return x, next_h, seqlen, logits

    def none_state(self):
        state = [None for _ in range(self.args.n_layer)]
        return state

    @staticmethod
    def from_original_pretrained(huggingface_model_id: str):
        '''
            for layer_name, tensor in state_dict.items():
                print(f"{layer_name}: {tensor.shape}")
            
            backbone.embedding.weight: torch.Size([50288, 2048])
            backbone.layers.0.norm.weight: torch.Size([2048])
            backbone.layers.0.mixer.dt_bias: torch.Size([64])
            backbone.layers.0.mixer.A_log: torch.Size([64])
            backbone.layers.0.mixer.D: torch.Size([64])
            backbone.layers.0.mixer.in_proj.weight: torch.Size([8512, 2048])
            backbone.layers.0.mixer.conv1d.weight: torch.Size([4352, 1, 4])
            backbone.layers.0.mixer.conv1d.bias: torch.Size([4352]
            backbone.layers.0.mixer.norm.weight: torch.Size([4096])
            backbone.layers.0.mixer.out_proj.weight: torch.Size([2048, 4096])
            ...
            backbone.norm_f.weight: torch.Size([2048])
            lm_head.weight: torch.Size([50288, 2048])
        '''
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        config_path = cached_file(huggingface_model_id, CONFIG_NAME)
        assert config_path, "Failed to get HuggingFace config file"
        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME)
        assert state_dict_path, "Failed to get HuggingFace state dict file"

        config = json.load(open(config_path))
        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config.get("pad_vocab_size_multiple", 16),
        )

        map_location = "cpu"
        state_dict = torch.load(
            state_dict_path, map_location=map_location, weights_only=True
        )

        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config.get("pad_vocab_size_multiple", 16),
        )
        mlh_model = Mamba2LMHeadModel(args=args)

        mlh_model_param = {}
        mlh_model_param['params'] = {}
        mlh_model_param['params']['embed'] = {
            'embedding': torch_to_jax(
                state_dict["backbone.embedding.weight"],
                dtype=jnp.float32),
        }
        for i in range(args.n_layer):

            norm_key = f"backbone.layers.{i}.norm.weight"
            temp_norm_param = {
                'weight': torch_to_jax(state_dict[norm_key]),
            }
            mlh_model_param['params'][f"norms_{i}"] = temp_norm_param

            key_prefix = f"backbone.layers.{i}.mixer."
            temp_layer_param = {
                'in_proj': {
                    'kernel': torch_to_jax(state_dict[f"{key_prefix}in_proj.weight"]).T,
                },
                'out_proj': {
                    'kernel': torch_to_jax(state_dict[f"{key_prefix}out_proj.weight"]).T,
                },
                #'A_log': torch_to_jax(state_dict[f"{key_prefix}A_log"]),
                'A_log': torch_to_jax(state_dict[f"{key_prefix}A_log"], dtype=jnp.float32), # this more likely fits PyTorch implementation param
                'D': torch_to_jax(state_dict[f"{key_prefix}D"]),
                'dt_bias': torch_to_jax(state_dict[f"{key_prefix}dt_bias"]),

                # HF param is float16 that makes calculation problem on conv
                # but i found that the original code which use nn.Conv1d contains float32 conv weight in it,
                # so I think this may cause trivial problem
                'conv_weight': torch_to_jax(
                    state_dict[f"{key_prefix}conv1d.weight"],
                    dtype=jnp.float32), # (conv_dim, 1, kernel_size)
                'conv_bias': torch_to_jax(state_dict[f"{key_prefix}conv1d.bias"]), # (conv_dim,)

                'norm': {
                    'weight': torch_to_jax(state_dict[f"{key_prefix}norm.weight"]),
                }
            }
            mlh_model_param['params'][f"layers_{i}"] = temp_layer_param
        mlh_model_param['params']['norm_f'] = {
            'weight': torch_to_jax(state_dict["backbone.norm_f.weight"]),
        }
        mlh_model_param['params']['lm_head'] = {
            'kernel': torch_to_jax(state_dict["lm_head.weight"]).T,
        }

        return mlh_model, mlh_model_param

    def prepare_state(
        self,
        param,
        input_ids
    ):
        #prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)
        prefix = input_ids[:-1]  # 先頭から最後のトークン以外を取得
        #tokens = jnp.expand_dims(input_ids_jax[-1:], axis=0)  # 最後のトークンを拡張して次元を追加

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.
        n_chunked = (prefix.shape[0] // self.args.chunk_size) * self.args.chunk_size
        if n_chunked > 0:
            temp_prefix_jax = jnp.expand_dims(prefix[:n_chunked], axis=0)
            nstate = self.none_state()
            #_, h = self(temp_prefix_jax, False, None)
            _, h = self.apply(param,
                temp_prefix_jax,
                False,
                nstate)
        else:
            h = [
                InferenceCache.alloc(1, self.args)
                for _ in range(self.args.n_layer)
            ]

        for i in range(n_chunked, prefix.shape[0]):
            temp_prefix_jax = jnp.expand_dims(prefix[i : i+1], axis=0)
            _, h = self.apply(param,
                temp_prefix_jax,
                True,
                h)

        return h

    def generate_token(
        self,
        param,
        tokens,
        h,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ):
        out, h = self.apply(param,
            tokens,
            True,
            h)
        logits = out[0, -1]

        # tempreture scaling
        if temperature != 1.0:
            logits = logits / temperature

        if self.use_jax:
            probs = jax.nn.softmax(logits)
        else:
            probs = F.softmax(logits, dim=-1)
            probs = torch_to_jax(probs)

        # apply top_k, top_p choice algorithms
        if top_k > 0:
            top_k_values, _ = jax.lax.top_k(probs, top_k)
            threshold = top_k_values[-1]  # the last value becomes threashold
            indices_to_remove = probs < threshold
            probs = jnp.where(indices_to_remove, 0, probs)  # set probability 0 for eliminating tokens
        if top_p < 1.0:
            probs = jax_to_torch(probs)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            if self.use_jax:
                sorted_probs_jax = torch_to_jax(sorted_probs)
                cum_probs_jax = jnp.cumsum(sorted_probs_jax)
                cum_probs = jax_to_torch(cum_probs_jax)
            else:
                cum_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0  # set probability 0 for eliminating tokens

        # sampling the next token here (JAX implementation)
        #key = jax.random.PRNGKey(1)  # use an appropriate seed value
        #next_token = jax.random.choice(key, a=probs.shape[0], p=probs)
        #next_token = jnp.expand_dims(next_token, axis=0)
        #next_token = jax_to_torch(next_token)

        # sampling the next token here (Torch implementation to meet PyTorch Mamba2 sampling)
        probs = jax_to_torch(probs)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = torch_to_jax(next_token)

        return next_token, h

    def generate(
        self,
        param,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[InferenceCache]]]:
        input_ids_jax = torch_to_jax(input_ids)
        tokens = jnp.expand_dims(input_ids_jax[-1:], axis=0)  # 最後のトークンを拡張して次元を追加

        # Prepare
        h = self.prepare_state(
            param,
            input_ids_jax)

        # Generate
        for _ in range(max_new_length):
            next_token, h = self.generate_token(
                param,
                tokens=tokens,
                h=h,
                max_new_length=max_new_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if next_token.item() == eos_token_id:
                return
            tokens = jnp.expand_dims(next_token, axis=0)

            yield cast(int, next_token.item()), h


class Mamba2(nn_flax.Module):
    args: Mamba2Config
    use_jax: bool = True

    def setup(self):
        super().__init__()

        args = self.args
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        conv_dim = args.d_inner + 2 * args.d_state
        self.conv_dim = conv_dim

        self.in_proj = nn_flax.Dense(features=d_in_proj, use_bias=False)
        self.out_proj = nn_flax.Dense(features=args.d_model, use_bias=False)
        self.norm =  RMSNorm(d=args.d_inner) ########

        self.A_log = self.param(
            'A_log',
            lambda rng: jax.numpy.zeros((args.nheads,))
        )
        self.D = self.param(
            'D',
            lambda rng: jax.numpy.zeros((args.nheads,))
        )
        self.dt_bias = self.param(
            'dt_bias',
            lambda rng: jax.numpy.zeros((args.nheads,))
        )

        self.conv_weight = self.param('conv_weight',
            lambda rng: jax.numpy.zeros((conv_dim, 1, args.d_conv))
        )  # (conv_dim, 1, kernel_size)
        self.conv_bias = self.param('conv_bias',
            lambda rng: jax.numpy.zeros((conv_dim,))
        )  # (conv_dim, 1, kernel_size)
        '''
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )
        '''

    def __call__(
        self,
        u: jnp.ndarray,
        step_mode: bool,
        h: InferenceCache | None = None,
    ):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        #if h is not None:
        if step_mode == True:
            return self.step(u, h)

        # in_proj
        zxbcdt = self.in_proj(u)

        split_sizes = [self.args.d_inner, self.args.d_inner + 2 * self.args.d_state, self.args.nheads]
        z, xBC, dt = split_tensor(zxbcdt, split_sizes, dim=-1, use_jax=self.use_jax)

        dt_bias = self.get_variable('params', 'dt_bias')
        dt = jax.nn.softplus(dt + dt_bias)

        if self.use_jax:
            # Calculate padding amount
            pad_amount = self.args.d_conv - u.shape[1]
            if pad_amount > 0:
                # Positive padding: pad the tensor
                conv_state = rearrange(xBC, 'b l d -> b d l')  # (batch, conv_dim, seqlen)
                conv_state_padded = jnp.pad(conv_state, ((0, 0), (0, 0), (pad_amount, 0)), mode='constant')
                conv_state = conv_state_padded[..., -self.args.d_conv:]  # Keep last d_conv elements
            elif pad_amount < 0:
                # Negative padding: slice the tensor
                conv_state = rearrange(xBC, 'b l d -> b d l')  # (batch, conv_dim, seqlen)
                conv_state = conv_state[..., -self.args.d_conv:]  # Keep last d_conv elements
            else:
                # Zero padding: use the tensor as is
                conv_state = rearrange(xBC, 'b l d -> b d l')[..., -self.args.d_conv:]
        else:
            # PyTorch implementation (as in the original code)
            xBC_torch = jax_to_torch(xBC)
            conv_state = F.pad(
                rearrange(xBC_torch, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
            )
            conv_state = torch_to_jax(conv_state)

        if self.use_jax:
            # Prepare convolution weights
            conv_weight = self.get_variable('params', 'conv_weight') # (conv_dim, 1, kernel_size)
            conv_bias = self.get_variable('params', 'conv_bias') ################################
            conv_weight = conv_weight.squeeze(1).T[:, None, :]  # (kernel_size, 1, conv_dim)
    
            # Perform depthwise convolution using xBC directly
            def depthwise_conv(inputs, filters):
                outputs = jax.lax.conv_general_dilated(
                    inputs,
                    filters,
                    window_strides=(1,),
                    padding=[(self.args.d_conv - 1, 0)],  # Left padding
                    dimension_numbers=('NWC', 'WIO', 'NWC'),
                    feature_group_count=self.conv_dim,
                )
                return outputs

            xBC_conv = depthwise_conv(xBC, conv_weight)  # (batch, seqlen, conv_dim)
            xBC_conv += conv_bias
    
            xBC = silu(xBC_conv)
        else:
            xBC_torch = jax_to_torch(xBC)
            xBC = silu(
                torch_to_jax(self.conv1d(xBC_torch.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :])
            )  # (batch, seqlen, d_inner + 2 * d_state))

        split_sizes = [self.args.d_inner, self.args.d_state, self.args.d_state]
        x, B, C = split_tensor(xBC, split_sizes, dim=-1, use_jax=self.use_jax)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)

        A_log = self.get_variable('params', 'A_log')
        A = -jnp.exp(A_log)  # JAX computation # (nheads,)

        y, ssm_state = ssd(
            x * jnp.expand_dims(dt, axis=-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
        )

        D = self.get_variable('params', 'D')
        y = y + jnp.expand_dims(D, axis=-1) * x

        y = rearrange(y, "b l h p -> b l (h p)")

        y = self.norm(y, z)

        # out_proj
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)

        return y, h

    def step(
        self,
        u: jnp.ndarray,
        h: InferenceCache
    ) -> tuple[Tensor, InferenceCache]:
        """
        Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) do not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        # in_proj
        zxbcdt = self.in_proj(u.squeeze(1))

        split_sizes = [self.args.d_inner, self.args.d_inner + 2 * self.args.d_state, self.args.nheads]
        z, xBC, dt = split_tensor(zxbcdt, split_sizes, dim=-1, use_jax=self.use_jax)

        # Advance convolution input
        rolled_conv_state = jnp.roll(h.conv_state, shift=-1, axis=-1)
        updated_conv_state = rolled_conv_state.at[:, :, -1].set(xBC)
        h = InferenceCache(conv_state=updated_conv_state, ssm_state=h.ssm_state)

        ###############
        # Convolution step
        conv_weight = self.get_variable('params', 'conv_weight') # (conv_dim, 1, kernel_size)
        h_conv_state = h.conv_state
        conv_weight = rearrange(conv_weight, "d 1 w -> d w")
        xBC = jnp.sum(h_conv_state * conv_weight, axis=-1)

        conv_bias = self.get_variable('params', 'conv_bias') ################################
        xBC += conv_bias

        xBC = silu(xBC)

        split_sizes = [self.args.d_inner, self.args.d_state, self.args.d_state]
        x, B, C = split_tensor(xBC, split_sizes, dim=-1, use_jax=self.use_jax)

        # Calculate A
        A_log = self.get_variable('params', 'A_log')
        A = -jnp.exp(A_log)  # JAX computation # (nheads,)

        ###############
        # SSM step
        dt_bias = self.get_variable('params', 'dt_bias')
        dt = jax.nn.softplus(dt + dt_bias)

        # dA calculation
        dA = jnp.exp(dt * A)

        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)

        dBx = jnp.einsum("bh, bn, bhp -> bhpn", dt, B, x)

        # update ssm_state in JAX
        ssm_state = h.ssm_state
        updated_ssm_state = ssm_state *  rearrange(dA, "b h -> b h 1 1") + dBx
        # instead of substituting h.ssm_state, generate new h
        h = InferenceCache(conv_state=h.conv_state, ssm_state=updated_ssm_state)

        # einsum in JAX
        y = jnp.einsum("bhpn, bn -> bhp", updated_ssm_state, C)

        D = self.get_variable('params', 'D')
        y = y + rearrange(D, "h -> h 1") * x

        y = rearrange(y, "b h p -> b (h p)")

        y = self.norm(y, z)

        # out_proj
        y = self.out_proj(y)

        y = jnp.expand_dims(y, axis=1)

        return y, h

def segsum(
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.shape[-1]  # size for the dimension of the "time"

    # JAX's repeat process
    x = repeat(x, "... d -> ... d e", e=T)
    #x = jnp.tile(x[..., None], (1, 1, T))
    # generate a lower triangular matrix mask in JAX
    mask_jax = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    # JAX's masked_fill process
    x = jnp.where(mask_jax, x, 0)
    # JAX's cumulative sum (cumsum)
    x_segsum = jnp.cumsum(x, axis=-2)
    # apply the mask again (to only reserve the lower triangular parts)
    mask_jax = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum = jnp.where(mask_jax, x_segsum, -jnp.inf)

    return x_segsum


from typing import Optional, Tuple
def ssd(
    x: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    chunk_size: int,
    initial_states: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Structured State Space Duality (SSD) using JAX for internal computations.

    Arguments:
        x: (batch, seqlen, n_heads, d_head) - Input tensor
        A: (batch, seqlen, n_heads) - A tensor
        B: (batch, seqlen, n_heads, d_state) - B tensor
        C: (batch, seqlen, n_heads, d_state) - C tensor
        chunk_size: int - Size of each chunk
        initial_states: Optional[(batch, 1, n_heads, d_state, ...)] - Initial states

    Returns:
        Y: (batch, seqlen, n_heads, d_head) - Output tensor
        final_state: (batch, n_heads, d_state, ...) - Final state tensor
    """
    assert x.shape[1] % chunk_size == 0, "Sequence length must be divisible by chunk_size."

    # Rearrange into chunks using JAX's equivalent of einops.rearrange
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    # Rearrange A for cumulative sum
    A = rearrange(A, "b c l h -> b h c l")

    A_cumsum = jnp.cumsum(A, axis=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = jnp.exp(segsum(x=A))

    # Perform Einstein summation using JAX
    Y_diag = jnp.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)

    states = jnp.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1])

    states = jnp.concatenate([initial_states, states], axis=1)

    # Compute decay_chunk
    A_cumsum_padded = jnp.pad(A_cumsum[:, :, :, -1], ((0,0), (0,0), (1,0)), mode='constant')
    decay_chunk = jnp.exp(segsum(x=A_cumsum_padded))

    new_states = jnp.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)

    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    state_decay_out = jnp.exp(A_cumsum)

    Y_off = jnp.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = Y_diag + Y_off
    Y = rearrange(Y, "b c l h p -> b (c l) h p")

    return Y, final_state



class RMSNorm(nn_flax.Module):
    d: int
    eps: float = 1e-5

    def setup(self):
        """initialize the parameters"""
        super().__init__()

        self.weight = self.param('weight', lambda rng: jax.numpy.ones((self.d,)))

    def __call__(self, x, z=None):
        """
        Arguments:
            x: (batch, seqlen, d_model) input tensor
            z: optional gating tensor for scaling input
        """
        # Gated scaling (if z exists)
        if z is not None:
            x = x * silu(z)

        # Root Mean Square Layer Normalization
        mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        rsqrt = jnp.sqrt(1 / (mean_sq + self.eps))
        x_normalized = x * rsqrt * self.weight


        return x_normalized

def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    This function supports both JAX and PyTorch computations depending on the use_jax flag.
    """
    return x * jax.nn.sigmoid(x)
