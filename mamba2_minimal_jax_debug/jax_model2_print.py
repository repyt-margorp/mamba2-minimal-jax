"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

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

Device: TypeAlias = str | torch.device | None

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
            # JAXでインデックスを生成
            indices = jnp.arange(start, end)
            # JAXのtakeを使用して分割
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
        print(f"[Config] Initialized Mamba2Config with d_model={self.d_model}, "
              f"n_layer={self.n_layer}, d_state={self.d_state}, d_conv={self.d_conv}, "
              f"expand={self.expand}, headdim={self.headdim}, chunk_size={self.chunk_size}, "
              f"vocab_size={self.vocab_size}, pad_vocab_size_multiple={self.pad_vocab_size_multiple}")
        print(f"[Config] Computed d_inner={self.d_inner}, nheads={self.nheads}")


@dataclass
class InferenceCache:
    conv_state: jnp.ndarray  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: jnp.ndarray   # (batch, nheads, headdim, d_state)

    @staticmethod
    #def alloc(batch_size: int, args: 'Mamba2Config') -> 'InferenceCache':
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None) -> 'InferenceCache':
        print(f"[InferenceCache] Allocating InferenceCache for batch size: {batch_size}")

        conv_state = jnp.zeros(
            (batch_size, args.d_inner + 2 * args.d_state, args.d_conv)
        )
        print(f"[InferenceCache] conv_state shape: {conv_state.shape}")
        print(f"[InferenceCache] conv_state sample values: {conv_state.flatten()[:5]}")

        ssm_state = jnp.zeros(
            (batch_size, args.nheads, args.headdim, args.d_state)
        )
        print(f"[InferenceCache] ssm_state shape: {ssm_state.shape}")
        print(f"[InferenceCache] ssm_state sample values: {ssm_state.flatten()[:5]}")

        return InferenceCache(conv_state, ssm_state)

def print_state_dict_shapes(state_dict):
    """
    PyTorchのstate_dictの各レイヤー名とテンソルの形状を表示する関数。
    
    Args:
    - state_dict (OrderedDict): PyTorchモデルのstate_dict。
    
    Returns:
    - None: 各レイヤーの名前と形状を標準出力に表示する。
    """
    print("\n==== PyTorch State Dict ====\n")
    for layer_name, tensor in state_dict.items():
        print(f"{layer_name}: {tensor.shape}")


class FlaxMamba2LMHeadModel(nn_flax.Module):
    args: Mamba2Config
    use_jax: bool = True

    def setup(self):
        args = self.args

        self.embed = nn_flax.Embed(num_embeddings=args.vocab_size, features=args.d_model)
        self.norms = [FlaxRMSNorm(args.d_model) for _ in range(args.n_layer)]
        self.layers = [FlaxMamba2(args, True) for _ in range(args.n_layer)]
        self.norm_f = FlaxRMSNorm(args.d_model)
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
        print(f"[Mamba2LMHeadModel] Forward pass input_ids shape: {input_ids.shape}")
        print(f"[Mamba2LMHeadModel] input_ids sample values: {input_ids.flatten()[:5]}")
        seqlen = input_ids.shape[1]

        x = self.embed(input_ids)
        print(f"[Mamba2LMHeadModel] Embedding output shape: {x.shape}")
        print(f"[Mamba2LMHeadModel] Embedding sample values: {x.flatten()[:5]}")

        next_h = [None for _ in range(self.args.n_layer)]

        for i in range(self.args.n_layer):
            print(f"[Mamba2LMHeadModel] Processing layer {i + 1}/{self.args.n_layer}")
            flax_norm = self.norms[i]
            flax_layer = self.layers[i]

            normed_x = flax_norm(x)
            #y, h[i] = flax_layer( # this fails
            y, next_h[i] = flax_layer(
                        normed_x,
                        step_mode,
                        h[i])
            print(f"  [Layer {i + 1}] Output shape after mixer: {y.shape}")
            print(f"  [Layer {i + 1}] Output sample values after mixer: {y.flatten()[:5]}")
            x = y + x
            print(f"  [Layer {i + 1}] Residual connection shape: {x.shape}")
            print(f"  [Layer {i + 1}] Residual connection sample values: {x.flatten()[:5]}")

        x = self.norm_f(x)
        print(f"[Mamba2LMHeadModel] Final backbone norm output shape: {x.shape}")
        print(f"[Mamba2LMHeadModel] Final backbone norm output sample values: {x.flatten()[:5]}")

        if self.use_jax:
            print("[Mamba2LMHeadModel] Using JAX for lm_head")
            logits = self.lm_head(x)
            #logits = jax_to_torch(logits)
        else:
            x_torch = jax_to_torch(x)
            print("[Mamba2LMHeadModel] Using PyTorch for lm_head")
            logits = self.torch_lm_head(x_torch)
            logits = torch_to_jax(logits)
        print(f"[Mamba2LMHeadModel] Logits shape: {logits.shape}")
        print(f"[Mamba2LMHeadModel] Logits sample values: {logits.flatten()[:5]}")

        return logits[:, :seqlen], cast(list[InferenceCache], next_h)
        #return x, h, seqlen, logits # this fails
        #return x, next_h, seqlen, logits

    def none_state(self):
        state = [None for _ in range(self.args.n_layer)]
        return state

    @staticmethod
    def from_original_pretrained(huggingface_model_id: str, device: Device = None):
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

        print(f"[Mamba2LMHeadModel] Loading pre-trained model from HuggingFace ID: {huggingface_model_id}")
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
        print("[FlaxMamba2LMHeadModel] State Dict:")
        print_state_dict_shapes(state_dict)
        print("[FlaxMamba2LMHeadModel] Loaded state dict.")

        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config.get("pad_vocab_size_multiple", 16),
        )
        mlh_model = FlaxMamba2LMHeadModel(args=args)

        mlh_model_param = {}
        mlh_model_param['params'] = {}
        mlh_model_param['params']['embed'] = {
            'embedding': torch_to_jax(
                state_dict["backbone.embedding.weight"],
                dtype=jnp.float32),
        }
        for i in range(args.n_layer):
            #print("[MlhModelParam]", mlh_model_param)
            #print("[ApplyParam]", flax_params)

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

        # 確率を基にtop_k, top_pを適用する
        if top_k > 0:
            top_k_values, _ = jax.lax.top_k(probs, top_k)
            threshold = top_k_values[-1]  # 最後の値がしきい値
            indices_to_remove = probs < threshold
            probs = jnp.where(indices_to_remove, 0, probs)  # 削除対象の確率を0に設定
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
            probs[indices_to_remove] = 0  # 削除対象の確率を0に設定

        # ここで次のトークンを確率に基づいてサンプリング
        #key = jax.random.PRNGKey(1)  # シードを適切に設定
        #next_token = jax.random.choice(key, a=probs.shape[0], p=probs)
        #next_token = jnp.expand_dims(next_token, axis=0)
        #next_token = jax_to_torch(next_token)

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

class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None, use_jax = True):
        super().__init__()
        self.args = args
        self.device = device
        self.use_jax = use_jax

        #self.mlh_model = FlaxMamba2LMHeadModel(args=self.args)
        #self.mlh_model_param = {}

    @staticmethod
    def from_pretrained(huggingface_model_id: str, device: Device = None):
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        print(f"[Mamba2LMHeadModel] Loading pre-trained model from HuggingFace ID: {huggingface_model_id}")
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

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, map_location=map_location, weights_only=True
        )
        print("[Mamba2LMHeadModel] State Dict:")
        print_state_dict_shapes(state_dict)
        print("[Mamba2LMHeadModel] Loaded state dict.")
        model = Mamba2LMHeadModel(args, device=device)
        #model.load_state_dict(state_dict)
        #model.eval()
        print("[Mamba2LMHeadModel] Model loaded and set to evaluation mode.")

        model.mlh_model, model.mlh_model_param = FlaxMamba2LMHeadModel.from_original_pretrained(huggingface_model_id)

        return model

    def forward(
        self, input_ids: LongTensor, step_mode: bool, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[LongTensor, list[InferenceCache]]:
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
        pass

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[InferenceCache]]]:
        return self.mlh_model.generate(
            self.mlh_model_param,
            input_ids=input_ids,
            max_new_length=max_new_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )


class FlaxMamba2(nn_flax.Module):
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
        self.norm =  FlaxRMSNorm(d=args.d_inner) ########

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
        print(f"  [FlaxMamba2] __call__")
        print(f"  [FlaxMamba2] u shape: {u.shape}")
        print(f"  [FlaxMamba2] u sample values: {u.flatten()[:5]}")
        #print(f"  [FlaxMamba2] h: {h}")

        print(f"  [FlaxMamba2] Boolean Branch: {h is not None}")
        print(f"  [FlaxMamba2] Step Mode: {step_mode}")

        #if h is not None:
        if step_mode == True:
            return self.step(u, h)

        # in_proj
        zxbcdt = self.in_proj(u)
        print(f"  [FlaxMamba2] in_proj output shape: {zxbcdt.shape}")
        print(f"  [FlaxMamba2] in_proj output sample values: {zxbcdt.flatten()[:5]}")

        split_sizes = [self.args.d_inner, self.args.d_inner + 2 * self.args.d_state, self.args.nheads]
        z, xBC, dt = split_tensor(zxbcdt, split_sizes, dim=-1, use_jax=self.use_jax)
        print(f"  [FlaxMamba2] Split shapes (step) -> z: {z.shape}, xBC: {xBC.shape}, dt: {dt.shape}")
        print(f"  [FlaxMamba2] z (step) sample values: {z.flatten()[:5]}")
        print(f"  [FlaxMamba2] xBC (step) sample values: {xBC.flatten()[:5]}")
        print(f"  [FlaxMamba2] dt (step) sample values: {dt.flatten()[:5]}")

        dt_bias = self.get_variable('params', 'dt_bias')
        dt = jax.nn.softplus(dt + dt_bias)
        print(f"  [FlaxMamba2] dt after softplus shape: {dt.shape}")
        print(f"  [FlaxMamba2] dt after softplus sample values: {dt.flatten()[:5]}")

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
            print(f"  [FlaxMamba2] conv_state shape after padding/slicing: {conv_state.shape}")
            print(f"  [FlaxMamba2] conv_state sample values: {conv_state.flatten()[:5]}")
        else:
            # PyTorch implementation (as in the original code)
            xBC_torch = jax_to_torch(xBC)
            conv_state = F.pad(
                rearrange(xBC_torch, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
            )
            conv_state = torch_to_jax(conv_state)
            print(f"  [FlaxMamba2] conv_state shape after padding: {conv_state.shape}")
            print(f"  [FlaxMamba2] conv_state after padding sample values: {conv_state.flatten()[:5]}")

        if self.use_jax:
            # Prepare convolution weights
            conv_weight = self.get_variable('params', 'conv_weight') # (conv_dim, 1, kernel_size)
            conv_bias = self.get_variable('params', 'conv_bias') ################################
            conv_weight = conv_weight.squeeze(1).T[:, None, :]  # (kernel_size, 1, conv_dim)
            print(f"  [FlaxMamba2] conv_weight shape: {conv_weight.shape}")
            print(f"  [FlaxMamba2] conv_weight sample values: {conv_weight.flatten()[:5]}")
    
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
            print(f"  [FlaxMamba2] Added conv1d bias shape: {xBC_conv.shape}")
            print(f"  [FlaxMamba2] xBC_conv after adding bias sample values: {xBC_conv.flatten()[:5]}")
    
            xBC = silu(xBC_conv)
            print(f"  [FlaxMamba2] xBC after convolution shape: {xBC.shape}")
            print(f"  [FlaxMamba2] xBC after convolution sample values: {xBC.flatten()[:5]}")
        else:
            xBC_torch = jax_to_torch(xBC)
            xBC = silu(
                torch_to_jax(self.conv1d(xBC_torch.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :])
            )  # (batch, seqlen, d_inner + 2 * d_state))
            print(f"  [FlaxMamba2] conv1d output shape: {xBC.shape}")
            print(f"  [FlaxMamba2] conv1d output sample values: {xBC.flatten()[:5]}")

        split_sizes = [self.args.d_inner, self.args.d_state, self.args.d_state]
        x, B, C = split_tensor(xBC, split_sizes, dim=-1, use_jax=self.use_jax)
        print(f"  [FlaxMamba2] Split conv1d output -> x: {x.shape}, B: {B.shape}, C: {C.shape}")
        print(f"  [FlaxMamba2] x sample values: {x.flatten()[:5]}")
        print(f"  [FlaxMamba2] B sample values: {B.flatten()[:5]}")
        print(f"  [FlaxMamba2] C sample values: {C.flatten()[:5]}")

        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        print(f"  [FlaxMamba2] x after rearrange shape: {x.shape}")
        print(f"  [FlaxMamba2] x after rearrange sample values: {x.flatten()[:5]}")

        A_log = self.get_variable('params', 'A_log')
        A = -jnp.exp(A_log)  # JAX computation # (nheads,)
        print(f"  [FlaxMamba2] A shape: {A.shape}")
        print(f"  [FlaxMamba2] A sample values: {A.flatten()[:5]}")

        y, ssm_state = ssd(
            x * jnp.expand_dims(dt, axis=-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=None,
        )
        print(f"  [FlaxMamba2] ssd output y shape: {y.shape}, ssm_state shape: {ssm_state.shape}")
        print(f"  [FlaxMamba2] y sample values: {y.flatten()[:5]}")
        print(f"  [FlaxMamba2] ssm_state sample values: {ssm_state.flatten()[:5]}")

        D = self.get_variable('params', 'D')
        y = y + jnp.expand_dims(D, axis=-1) * x
        print(f"  [FlaxMamba2] y after adding D scaling shape: {y.shape}")
        print(f"  [FlaxMamba2] y after adding D scaling sample values: {y.flatten()[:5]}")

        y = rearrange(y, "b l h p -> b l (h p)")
        print(f"  [FlaxMamba2] y after rearrange to (b, l, d_inner): {y.shape}")
        print(f"  [FlaxMamba2] y after rearrange sample values: {y.flatten()[:5]}")

        y = self.norm(y, z)
        print(f"  [FlaxMamba2] y after RMSNorm shape: {y.shape}")
        print(f"  [FlaxMamba2] y after RMSNorm sample values: {y.flatten()[:5]}")

        # out_proj
        y = self.out_proj(y)
        print(f"  [FlaxMamba2] y after out_proj shape: {y.shape}")
        print(f"  [FlaxMamba2] y after out_proj sample values: {y.flatten()[:5]}")

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
        print(f"  [FlaxMamba2] step")
        print(f"  [FlaxMamba2] u shape: {u.shape}")
        print(f"  [FlaxMamba2] u sample values: {u.flatten()[:5]}")
        #print(f"  [FlaxMamba2] h: {h}")

        assert u.shape[1] == 1, "Only one token can be decoded per inference step"
        print("[Mamba2] Performing single inference step.")

        # in_proj
        zxbcdt = self.in_proj(u.squeeze(1))
        print(f"  [FlaxMamba2] in_proj output shape: {zxbcdt.shape}")
        print(f"  [FlaxMamba2] in_proj output sample values: {zxbcdt.flatten()[:5]}")

        split_sizes = [self.args.d_inner, self.args.d_inner + 2 * self.args.d_state, self.args.nheads]
        z, xBC, dt = split_tensor(zxbcdt, split_sizes, dim=-1, use_jax=self.use_jax)
        print(f"  [FlaxMamba2] Split shapes (step) -> z: {z.shape}, xBC: {xBC.shape}, dt: {dt.shape}")
        print(f"  [FlaxMamba2] z (step) sample values: {z.flatten()[:5]}")
        print(f"  [FlaxMamba2] xBC (step) sample values: {xBC.flatten()[:5]}")
        print(f"  [FlaxMamba2] dt (step) sample values: {dt.flatten()[:5]}")

        # Advance convolution input
        rolled_conv_state = jnp.roll(h.conv_state, shift=-1, axis=-1)
        updated_conv_state = rolled_conv_state.at[:, :, -1].set(xBC)
        h = InferenceCache(conv_state=updated_conv_state, ssm_state=h.ssm_state)
        print(f"  [FlaxMamba2] conv_state updated shape: {h.conv_state.shape}")
        print(f"  [FlaxMamba2] conv_state updated sample values: {h.conv_state.flatten()[:5]}")

        ################
        # Convolution step
        ################
        print("[Mamba2] Using JAX for convolution and rearrange")
        conv_weight = self.get_variable('params', 'conv_weight') # (conv_dim, 1, kernel_size)
        h_conv_state = h.conv_state
        conv_weight = rearrange(conv_weight, "d 1 w -> d w")
        xBC = jnp.sum(h_conv_state * conv_weight, axis=-1)
        print(f"  [FlaxMamba2] xBC after convolution sum shape: {xBC.shape}")
        print(f"  [FlaxMamba2] xBC after convolution sum sample values: {xBC.flatten()[:5]}")

        conv_bias = self.get_variable('params', 'conv_bias') ################################
        xBC += conv_bias
        print(f"  [FlaxMamba2] Added conv1d bias shape: {xBC.shape}")
        print(f"  [FlaxMamba2] xBC after adding bias sample values: {xBC.flatten()[:5]}")

        xBC = silu(xBC)
        print(f"  [FlaxMamba2] xBC after silu shape: {xBC.shape}")
        print(f"  [FlaxMamba2] xBC after silu sample values: {xBC.flatten()[:5]}")

        split_sizes = [self.args.d_inner, self.args.d_state, self.args.d_state]
        x, B, C = split_tensor(xBC, split_sizes, dim=-1, use_jax=self.use_jax)
        print(f"  [FlaxMamba2] Split xBC -> x: {x.shape}, B: {B.shape}, C: {C.shape}")
        print(f"  [FlaxMamba2] x (step) sample values: {x.flatten()[:5]}")
        print(f"  [FlaxMamba2] B (step) sample values: {B.flatten()[:5]}")
        print(f"  [FlaxMamba2] C (step) sample values: {C.flatten()[:5]}")

        # Aの計算
        A_log = self.get_variable('params', 'A_log')
        A = -jnp.exp(A_log)  # JAX computation # (nheads,)
        print(f"  [FlaxMamba2] A shape: {A.shape}")
        print(f"  [FlaxMamba2] A sample values: {A.flatten()[:5]}")

        ###############
        # SSM step
        ###############
        dt_bias = self.get_variable('params', 'dt_bias')
        dt = jax.nn.softplus(dt + dt_bias)  # JAXでsoftplus
        print(f"  [FlaxMamba2] dt after softplus (step) shape: {dt.shape}")
        print(f"  [FlaxMamba2] dt after softplus (step) sample values: {dt.flatten()[:5]}")

        # dAの計算
        dA = jnp.exp(dt * A)  # JAXでの計算
        print(f"  [FlaxMamba2] dA shape: {dA.shape}")
        print(f"  [FlaxMamba2] dA sample values: {dA.flatten()[:5]}")

        # rearrange 実行前の x の形状を確認
        print(f"[FlaxMamba2] Before rearrange, x shape: {x.shape}")
        print("[FlaxMamba2] Using JAX for rearrange")
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        print(f"  [FlaxMamba2] x after rearrange (step) shape: {x.shape}")
        print(f"  [FlaxMamba2] x after rearrange (step) sample values: {x.flatten()[:5]}")

        dBx = jnp.einsum("bh, bn, bhp -> bhpn", dt, B, x)  # JAXでの計算
        print(f"  [FlaxMamba2] dBx shape: {dBx.shape}")
        print(f"  [FlaxMamba2] dBx sample values: {dBx.flatten()[:5]}")
        print(f"  [FlaxMamba2] dBx shape: {dBx.shape}")
        print(f"  [FlaxMamba2] dBx sample values: {dBx.flatten()[:5]}")

        # JAXでのssm_state更新
        ssm_state = h.ssm_state
        updated_ssm_state = ssm_state *  rearrange(dA, "b h -> b h 1 1") + dBx
        # ここでh.ssm_stateに直接代入するのではなく、新しいhを生成
        h = InferenceCache(conv_state=h.conv_state, ssm_state=updated_ssm_state)  # 新しいhオブジェクトを生成
        print(f"  [FlaxMamba2] ssm_state updated shape: {h.ssm_state.shape}")
        print(f"  [FlaxMamba2] ssm_state updated sample values: {h.ssm_state.flatten()[:5]}")

        # JAXでのeinsum
        y = jnp.einsum("bhpn, bn -> bhp", updated_ssm_state, C)
        print(f"  [FlaxMamba2] y after einsum shape: {y.shape}")
        print(f"  [FlaxMamba2] y after einsum sample values: {y.flatten()[:5]}")

        D = self.get_variable('params', 'D')
        y = y + rearrange(D, "h -> h 1") * x
        print(f"  [FlaxMamba2] y after adding D scaling (step) shape: {y.shape}")
        print(f"  [FlaxMamba2] y after adding D scaling (step) sample values: {y.flatten()[:5]}")

        y = rearrange(y, "b h p -> b (h p)")
        print(f"  [FlaxMamba2] y after rearrange to (b, d_inner) shape: {y.shape}")
        print(f"  [FlaxMamba2] y after rearrange to (b, d_inner) sample values: {y.flatten()[:5]}")

        # RMSNormの処理は既存のコードをそのまま使用
        y = self.norm(y, z)
        print(f"  [FlaxMamba2] y after RMSNorm (step) shape: {y.shape}")
        print(f"  [FlaxMamba2] y after RMSNorm (step) sample values: {y.flatten()[:5]}")

        # out_proj
        y = self.out_proj(y)
        print(f"  [FlaxMamba2] y after out_proj (step) shape: {y.shape}")
        print(f"  [FlaxMamba2] y after out_proj (step) sample values: {y.flatten()[:5]}")

        y = jnp.expand_dims(y, axis=1)
        print(f"  [FlaxMamba2] y after expand_dims (step) shape: {y.shape}")
        print(f"  [FlaxMamba2] y after expand_dims (step) sample values: {y.flatten()[:5]}")

        return y, h

def segsum(
    x: jnp.ndarray,
    device: Device = None
) -> jnp.ndarray:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.shape[-1]  # 時間次元のサイズ

    # JAXでrepeat相当の処理
    x = repeat(x, "... d -> ... d e", e=T)
    #x = jnp.tile(x[..., None], (1, 1, T))
    # 下三角行列のマスクをJAXで作成
    mask_jax = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    # JAXでmasked_fill相当の処理
    x = jnp.where(mask_jax, x, 0)
    # JAXで累積和 (cumsum)
    x_segsum = jnp.cumsum(x, axis=-2)
    # 再度マスクを適用 (下三角部分のみ残す)
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
    device: Optional[torch.device] = None
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
        device: Optional[torch.device] - Device to place the output tensors

    Returns:
        Y: (batch, seqlen, n_heads, d_head) - Output tensor
        final_state: (batch, n_heads, d_state, ...) - Final state tensor
    """
    print("[ssd_jax] Starting SSD computation with JAX...")
    assert x.shape[1] % chunk_size == 0, "Sequence length must be divisible by chunk_size."

    # Rearrange into chunks using JAX's equivalent of einops.rearrange
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]
    print(f"  [ssd_jax] After chunking shapes -> x: {x.shape}, A: {A.shape}, B: {B.shape}, C: {C.shape}")
    print(f"  [ssd_jax] x after chunking sample values: {x.flatten()[:5]}")
    print(f"  [ssd_jax] A after chunking sample values: {A.flatten()[:5]}")
    print(f"  [ssd_jax] B after chunking sample values: {B.flatten()[:5]}")
    print(f"  [ssd_jax] C after chunking sample values: {C.flatten()[:5]}")

    # Rearrange A for cumulative sum
    A = rearrange(A, "b c l h -> b h c l")
    print(f"  [ssd_jax] A rearranged shape: {A.shape}")
    print(f"  [ssd_jax] A rearranged sample values: {A.flatten()[:5]}")

    A_cumsum = jnp.cumsum(A, axis=-1)
    print(f"  [ssd_jax] A_cumsum shape: {A_cumsum.shape}")
    print(f"  [ssd_jax] A_cumsum sample values: {A_cumsum.flatten()[:5]}")

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = jnp.exp(segsum(x=A, device=device))
    print(f"  [ssd_jax] L shape: {L.shape}")
    print(f"  [ssd_jax] L sample values: {L.flatten()[:5]}")

    # Perform Einstein summation using JAX
    Y_diag = jnp.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)
    print(f"  [ssd_jax] Y_diag shape: {Y_diag.shape}")
    print(f"  [ssd_jax] Y_diag sample values: {Y_diag.flatten()[:5]}")

    # 2. Compute the state for each intra-chunk
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
    print(f"  [ssd_jax] decay_states shape: {decay_states.shape}")
    print(f"  [ssd_jax] decay_states sample values: {decay_states.flatten()[:5]}")

    states = jnp.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)
    print(f"  [ssd_jax] states shape: {states.shape}")
    print(f"  [ssd_jax] states sample values: {states.flatten()[:5]}")

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1])
        print("  [ssd_jax] Initialized initial_states with zeros.")
        print(f"  [ssd_jax] initial_states sample values: {initial_states.flatten()[:5]}")

    states = jnp.concatenate([initial_states, states], axis=1)
    print(f"  [ssd_jax] states after concatenation shape: {states.shape}")
    print(f"  [ssd_jax] states after concatenation sample values: {states.flatten()[:5]}")

    # Compute decay_chunk
    A_cumsum_padded = jnp.pad(A_cumsum[:, :, :, -1], ((0,0), (0,0), (1,0)), mode='constant')
    decay_chunk = jnp.exp(segsum(x=A_cumsum_padded, device=device))
    print(f"  [ssd_jax] decay_chunk shape: {decay_chunk.shape}")
    print(f"  [ssd_jax] decay_chunk sample values: {decay_chunk.flatten()[:5]}")

    new_states = jnp.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    print(f"  [ssd_jax] new_states shape: {new_states.shape}")
    print(f"  [ssd_jax] new_states sample values: {new_states.flatten()[:5]}")

    states, final_state = new_states[:, :-1], new_states[:, -1]
    print(f"  [ssd_jax] states shape after splitting: {states.shape}, final_state shape: {final_state.shape}")
    print(f"  [ssd_jax] states after splitting sample values: {states.flatten()[:5]}")
    print(f"  [ssd_jax] final_state sample values: {final_state.flatten()[:5]}")

    # 4. Compute state -> output conversion per chunk
    state_decay_out = jnp.exp(A_cumsum)
    print(f"  [ssd_jax] state_decay_out shape: {state_decay_out.shape}")
    print(f"  [ssd_jax] state_decay_out sample values: {state_decay_out.flatten()[:5]}")

    Y_off = jnp.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    print(f"  [ssd_jax] Y_off shape: {Y_off.shape}")
    print(f"  [ssd_jax] Y_off sample values: {Y_off.flatten()[:5]}")

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = Y_diag + Y_off
    Y = rearrange(Y, "b c l h p -> b (c l) h p")
    print(f"  [ssd_jax] Y after adding Y_diag and Y_off shape: {Y.shape}")
    print(f"  [ssd_jax] Y after adding Y_diag and Y_off sample values: {Y.flatten()[:5]}")

    return Y, final_state



class FlaxRMSNorm(nn_flax.Module):
    d: int
    eps: float = 1e-5

    def setup(self):
        """パラメータの初期化"""
        super().__init__()

        self.weight = self.param('weight', lambda rng: jax.numpy.ones((self.d,)))

    def __call__(self, x, z=None):
        """
        Arguments:
            x: (batch, seqlen, d_model) input tensor
            z: optional gating tensor for scaling input
        """
        # Gated scaling (zが存在する場合)
        if z is not None:
            x = x * silu(z)
            print(f"[FlaxRMSNorm] x after gated scaling shape: {x.shape}")
            print(f"[FlaxRMSNorm] x after gated scaling sample values: {x.flatten()[:5]}")

        # Root Mean Square Layer Normalization
        mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        rsqrt = jnp.sqrt(1 / (mean_sq + self.eps))
        x_normalized = x * rsqrt * self.weight

        print(f"[FlaxRMSNorm] x_normalized shape: {x_normalized.shape}")
        print(f"[FlaxRMSNorm] x_normalized sample values: {x_normalized.flatten()[:5]}")

        return x_normalized

def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    This function supports both JAX and PyTorch computations depending on the use_jax flag.
    """
    return x * jax.nn.sigmoid(x)
