"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch with enhanced
debugging capabilities to print tensor shapes and sample values during execution.

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


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        print(f"[InferenceCache] Allocating InferenceCache for batch size: {batch_size}")
        conv_state = torch.zeros(
            batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
        )
        print(f"[InferenceCache] conv_state shape: {conv_state.shape}")
        print(f"[InferenceCache] conv_state sample values: {conv_state.flatten()[:5]}")
        ssm_state = torch.zeros(
            batch_size, args.nheads, args.headdim, args.d_state, device=device
        )
        print(f"[InferenceCache] ssm_state shape: {ssm_state.shape}")
        print(f"[InferenceCache] ssm_state sample values: {ssm_state.flatten()[:5]}")
        return InferenceCache(conv_state, ssm_state)


class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        print("[Mamba2LMHeadModel] Initializing backbone...")
        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        print(f"[Mamba2LMHeadModel] Initialized embedding with vocab_size={args.vocab_size} and d_model={args.d_model}")
        for i, layer in enumerate(self.backbone.layers):
            print(f"[Mamba2LMHeadModel] Initialized layer {i + 1}/{self.args.n_layer}")

        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        self.lm_head.weight = self.backbone.embedding.weight
        print("[Mamba2LMHeadModel] Initialized LM Head and tied weights with embedding.")

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
        print("[Mamba2LMHeadModel] Loaded state dict.")
        model = Mamba2LMHeadModel(args, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("[Mamba2LMHeadModel] Model loaded and set to evaluation mode.")

        # Print shapes of some key parameters for debugging
        print("[Mamba2LMHeadModel] Loaded model parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")
            # Print sample values
            if param.numel() > 0:
                print(f"    Sample values from {name}: {param.flatten()[:5]}")
            break  # Remove this break to see more parameters

        return model

    def forward(
        self, input_ids: LongTensor, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[LongTensor, list[InferenceCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) tokens from `EleutherAI/gpt-neox-20b` tokenizer
            h: hidden states for inference step. If present the constant-time
               (wrt sequence length) inference path will be taken, input_ids
               should have shape (batch, 1) containing the next batch of prompt
               token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing `input_ids`
        """
        print(f"[Mamba2LMHeadModel] Forward pass input_ids shape: {input_ids.shape}")
        print(f"[Mamba2LMHeadModel] input_ids sample values: {input_ids.flatten()[:5]}")
        seqlen = input_ids.shape[1]

        if h is None:
            h = [None for _ in range(self.args.n_layer)]
            print("[Mamba2LMHeadModel] Initialized hidden states.")

        x = self.backbone.embedding(input_ids)
        print(f"[Mamba2LMHeadModel] Embedding output shape: {x.shape}")
        print(f"[Mamba2LMHeadModel] Embedding sample values: {x.flatten()[:5]}")

        for i, layer in enumerate(self.backbone.layers):
            print(f"[Mamba2LMHeadModel] Processing layer {i + 1}/{self.args.n_layer}")
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            print(f"  [Layer {i + 1}] Output shape after mixer: {y.shape}")
            print(f"  [Layer {i + 1}] Output sample values after mixer: {y.flatten()[:5]}")
            x = y + x
            print(f"  [Layer {i + 1}] Residual connection shape: {x.shape}")
            print(f"  [Layer {i + 1}] Residual connection sample values: {x.flatten()[:5]}")

        x = self.backbone.norm_f(x)
        print(f"[Mamba2LMHeadModel] Final backbone norm output shape: {x.shape}")
        print(f"[Mamba2LMHeadModel] Final backbone norm output sample values: {x.flatten()[:5]}")
        logits = self.lm_head(x)
        print(f"[Mamba2LMHeadModel] Logits shape: {logits.shape}")
        print(f"[Mamba2LMHeadModel] Logits sample values: {logits.flatten()[:5]}")

        return logits[:, :seqlen], cast(list[InferenceCache], h)

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[InferenceCache]]]:
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.
        n_chunked = (prefix.shape[0] // self.args.chunk_size) * self.args.chunk_size
        if n_chunked > 0:
            _, h = self(prefix[:n_chunked].unsqueeze(0), None)
        else:
            h = [
                InferenceCache.alloc(1, self.args, device=self.device)
                for _ in range(self.args.n_layer)
            ]
        for i in range(n_chunked, prefix.shape[0]):
            _, h = self(prefix[i : i + 1].unsqueeze(0), h)

        # Generate
        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self(tokens, h)
            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = next_token.unsqueeze(0)
            yield cast(int, next_token.item()), h


class Mamba2(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        print("[Mamba2] Initializing Mamba2 mixer...")
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        print(f"  [Mamba2] d_in_proj: {d_in_proj}")
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)
        print(f"  [Mamba2] Initialized in_proj with input_dim={args.d_model}, output_dim={d_in_proj}")
        print(f"  [Mamba2] in_proj.weight shape: {self.in_proj.weight.shape}")
        print(f"  [Mamba2] in_proj.weight sample values: {self.in_proj.weight.flatten()[:5]}")

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )
        print(f"  [Mamba2] Initialized conv1d with in_channels={conv_dim}, "
              f"out_channels={conv_dim}, kernel_size={args.d_conv}")
        print(f"  [Mamba2] conv1d.weight shape: {self.conv1d.weight.shape}")
        print(f"  [Mamba2] conv1d.weight sample values: {self.conv1d.weight.flatten()[:5]}")

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        print(f"  [Mamba2] Initialized dt_bias with shape: {self.dt_bias.shape}")
        print(f"  [Mamba2] dt_bias sample values: {self.dt_bias.flatten()[:5]}")
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        print(f"  [Mamba2] Initialized A_log with shape: {self.A_log.shape}")
        print(f"  [Mamba2] A_log sample values: {self.A_log.flatten()[:5]}")
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        print(f"  [Mamba2] Initialized D with shape: {self.D.shape}")
        print(f"  [Mamba2] D sample values: {self.D.flatten()[:5]}")
        print("  [Mamba2] Initialized A_log and D parameters.")

        self.norm = RMSNorm(args.d_inner, device=device)
        print("  [Mamba2] Initialized RMSNorm.")

        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)
        print("  [Mamba2] Initialized out_proj.")
        print(f"  [Mamba2] out_proj.weight shape: {self.out_proj.weight.shape}")
        print(f"  [Mamba2] out_proj.weight sample values: {self.out_proj.weight.flatten()[:5]}")

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        print("[Mamba2] Initializing parameters with normal distribution.")
        nn.init.normal_(self.dt_bias, mean=0.0, std=0.02)
        print(f"  [Mamba2] dt_bias after init: {self.dt_bias.flatten()[:5]}")
        nn.init.normal_(self.A_log, mean=0.0, std=0.02)
        print(f"  [Mamba2] A_log after init: {self.A_log.flatten()[:5]}")
        nn.init.normal_(self.D, mean=0.0, std=0.02)
        print(f"  [Mamba2] D after init: {self.D.flatten()[:5]}")
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        print(f"  [Mamba2] in_proj.weight after init: {self.in_proj.weight.flatten()[:5]}")
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.02)
        print(f"  [Mamba2] conv1d.weight after init: {self.conv1d.weight.flatten()[:5]}")
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        print(f"  [Mamba2] out_proj.weight after init: {self.out_proj.weight.flatten()[:5]}")
        print("[Mamba2] Parameters initialized.")

    def forward(self, u: Tensor, h: InferenceCache | None = None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model)
            h: updated inference cache after processing `u`
        """
        if h:
            print("[Mamba2] Performing inference step.")
            return self.step(u, h)

        print("[Mamba2] Performing full forward pass.")
        A = -torch.exp(self.A_log)  # (nheads,)
        print(f"  [Mamba2] A shape: {A.shape}")
        print(f"  [Mamba2] A sample values: {A.flatten()[:5]}")
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        print(f"  [Mamba2] in_proj output shape: {zxbcdt.shape}")
        print(f"  [Mamba2] in_proj output sample values: {zxbcdt.flatten()[:5]}")

        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        print(f"  [Mamba2] Split shapes -> z: {z.shape}, xBC: {xBC.shape}, dt: {dt.shape}")
        print(f"  [Mamba2] z sample values: {z.flatten()[:5]}")
        print(f"  [Mamba2] xBC sample values: {xBC.flatten()[:5]}")
        print(f"  [Mamba2] dt sample values: {dt.flatten()[:5]}")

        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)
        print(f"  [Mamba2] dt after softplus shape: {dt.shape}")
        print(f"  [Mamba2] dt after softplus sample values: {dt.flatten()[:5]}")

        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )
        print(f"  [Mamba2] conv_state shape after padding: {conv_state.shape}")
        print(f"  [Mamba2] conv_state after padding sample values: {conv_state.flatten()[:5]}")

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        print(f"  [Mamba2] conv1d output shape: {xBC.shape}")
        print(f"  [Mamba2] conv1d output sample values: {xBC.flatten()[:5]}")

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        print(f"  [Mamba2] Split conv1d output -> x: {x.shape}, B: {B.shape}, C: {C.shape}")
        print(f"  [Mamba2] x sample values: {x.flatten()[:5]}")
        print(f"  [Mamba2] B sample values: {B.flatten()[:5]}")
        print(f"  [Mamba2] C sample values: {C.flatten()[:5]}")

        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        print(f"  [Mamba2] x after rearrange shape: {x.shape}")
        print(f"  [Mamba2] x after rearrange sample values: {x.flatten()[:5]}")

        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=self.device,
        )
        print(f"  [Mamba2] ssd output y shape: {y.shape}, ssm_state shape: {ssm_state.shape}")
        print(f"  [Mamba2] y sample values: {y.flatten()[:5]}")
        print(f"  [Mamba2] ssm_state sample values: {ssm_state.flatten()[:5]}")

        y = y + x * self.D.unsqueeze(-1)
        print(f"  [Mamba2] y after adding D scaling shape: {y.shape}")
        print(f"  [Mamba2] y after adding D scaling sample values: {y.flatten()[:5]}")

        y = rearrange(y, "b l h p -> b l (h p)")
        print(f"  [Mamba2] y after rearrange to (b, l, d_inner): {y.shape}")
        print(f"  [Mamba2] y after rearrange sample values: {y.flatten()[:5]}")

        y = self.norm(y, z)
        print(f"  [Mamba2] y after RMSNorm shape: {y.shape}")
        print(f"  [Mamba2] y after RMSNorm sample values: {y.flatten()[:5]}")

        y = self.out_proj(y)
        print(f"  [Mamba2] y after out_proj shape: {y.shape}")
        print(f"  [Mamba2] y after out_proj sample values: {y.flatten()[:5]}")

        h = InferenceCache(conv_state, ssm_state)
        print(f"  [Mamba2] Updated InferenceCache: conv_state shape {h.conv_state.shape}, ssm_state shape {h.ssm_state.shape}")
        print(f"  [Mamba2] conv_state sample values: {h.conv_state.flatten()[:5]}")
        print(f"  [Mamba2] ssm_state sample values: {h.ssm_state.flatten()[:5]}")

        return y, h

    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """Take a single inference step for the current input and hidden state

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
        print("[Mamba2] Performing single inference step.")

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        print(f"  [Mamba2] in_proj (step) output shape: {zxbcdt.shape}")
        print(f"  [Mamba2] in_proj (step) output sample values: {zxbcdt.flatten()[:5]}")

        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        print(f"  [Mamba2] Split shapes (step) -> z: {z.shape}, xBC: {xBC.shape}, dt: {dt.shape}")
        print(f"  [Mamba2] z (step) sample values: {z.flatten()[:5]}")
        print(f"  [Mamba2] xBC (step) sample values: {xBC.flatten()[:5]}")
        print(f"  [Mamba2] dt (step) sample values: {dt.flatten()[:5]}")

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        print(f"  [Mamba2] conv_state updated shape: {h.conv_state.shape}")
        print(f"  [Mamba2] conv_state updated sample values: {h.conv_state.flatten()[:5]}")

        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        print(f"  [Mamba2] xBC after convolution sum shape: {xBC.shape}")
        print(f"  [Mamba2] xBC after convolution sum sample values: {xBC.flatten()[:5]}")

        if self.conv1d.bias is not None:
            xBC += self.conv1d.bias
            print(f"  [Mamba2] Added conv1d bias shape: {xBC.shape}")
            print(f"  [Mamba2] xBC after adding bias sample values: {xBC.flatten()[:5]}")

        xBC = silu(xBC)
        print(f"  [Mamba2] xBC after silu shape: {xBC.shape}")
        print(f"  [Mamba2] xBC after silu sample values: {xBC.flatten()[:5]}")

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        print(f"  [Mamba2] Split xBC -> x: {x.shape}, B: {B.shape}, C: {C.shape}")
        print(f"  [Mamba2] x (step) sample values: {x.flatten()[:5]}")
        print(f"  [Mamba2] B (step) sample values: {B.flatten()[:5]}")
        print(f"  [Mamba2] C (step) sample values: {C.flatten()[:5]}")

        A = -torch.exp(self.A_log)  # (nheads,)
        print(f"  [Mamba2] A shape: {A.shape}")
        print(f"  [Mamba2] A sample values: {A.flatten()[:5]}")

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        print(f"  [Mamba2] dt after softplus (step) shape: {dt.shape}")
        print(f"  [Mamba2] dt after softplus (step) sample values: {dt.flatten()[:5]}")

        dA = torch.exp(dt * A)  # (batch, nheads)
        print(f"  [Mamba2] dA shape: {dA.shape}")
        print(f"  [Mamba2] dA sample values: {dA.flatten()[:5]}")

        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        print(f"  [Mamba2] x after rearrange (step) shape: {x.shape}")
        print(f"  [Mamba2] x after rearrange (step) sample values: {x.flatten()[:5]}")

        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        print(f"  [Mamba2] dBx shape: {dBx.shape}")
        print(f"  [Mamba2] dBx sample values: {dBx.flatten()[:5]}")

        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        print(f"  [Mamba2] ssm_state updated shape: {h.ssm_state.shape}")
        print(f"  [Mamba2] ssm_state updated sample values: {h.ssm_state.flatten()[:5]}")

        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        print(f"  [Mamba2] y after einsum shape: {y.shape}")
        print(f"  [Mamba2] y after einsum sample values: {y.flatten()[:5]}")

        y = y + rearrange(self.D, "h -> h 1") * x
        print(f"  [Mamba2] y after adding D scaling (step) shape: {y.shape}")
        print(f"  [Mamba2] y after adding D scaling (step) sample values: {y.flatten()[:5]}")

        y = rearrange(y, "b h p -> b (h p)")
        print(f"  [Mamba2] y after rearrange to (b, d_inner) shape: {y.shape}")
        print(f"  [Mamba2] y after rearrange to (b, d_inner) sample values: {y.flatten()[:5]}")

        y = self.norm(y, z)
        print(f"  [Mamba2] y after RMSNorm (step) shape: {y.shape}")
        print(f"  [Mamba2] y after RMSNorm (step) sample values: {y.flatten()[:5]}")

        y = self.out_proj(y)
        print(f"  [Mamba2] y after out_proj (step) shape: {y.shape}")
        print(f"  [Mamba2] y after out_proj (step) sample values: {y.flatten()[:5]}")

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    print(f"[segsum] Starting segsum computation with input shape: {x.shape}")
    print(f"  [segsum] input sample values: {x.flatten()[:5]}")
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    print(f"  [segsum] After repeating: {x.shape}")
    print(f"  [segsum] x after repeating sample values: {x.flatten()[:5]}")
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    print(f"  [segsum] Mask applied, shape: {x.shape}")
    print(f"  [segsum] x after masking sample values: {x.flatten()[:5]}")
    x_segsum = torch.cumsum(x, dim=-2)
    print(f"  [segsum] Cumulative sum and masking applied, shape: {x_segsum.shape}")
    print(f"  [segsum] x_segsum sample values: {x_segsum.flatten()[:5]}")
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    print(f"  [segsum] Final masked x_segsum shape: {x_segsum.shape}")
    print(f"  [segsum] Final masked x_segsum sample values: {x_segsum.flatten()[:5]}")
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structured State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    print("[ssd] Starting SSD computation...")
    assert x.shape[1] % chunk_size == 0, "Sequence length must be divisible by chunk_size."

    # Rearrange into chunks
    print("[ssd] Rearranging tensors into chunks.")
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]
    print(f"  [ssd] After chunking shapes -> x: {x.shape}, A: {A.shape}, B: {B.shape}, C: {C.shape}")
    print(f"  [ssd] x after chunking sample values: {x.flatten()[:5]}")
    print(f"  [ssd] A after chunking sample values: {A.flatten()[:5]}")
    print(f"  [ssd] B after chunking sample values: {B.flatten()[:5]}")
    print(f"  [ssd] C after chunking sample values: {C.flatten()[:5]}")

    A = rearrange(A, "b c l h -> b h c l")
    print(f"  [ssd] A rearranged shape: {A.shape}")
    print(f"  [ssd] A rearranged sample values: {A.flatten()[:5]}")
    A_cumsum = torch.cumsum(A, dim=-1)
    print(f"  [ssd] A_cumsum shape: {A_cumsum.shape}")
    print(f"  [ssd] A_cumsum sample values: {A_cumsum.flatten()[:5]}")

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    print(f"  [ssd] L shape: {L.shape}")
    print(f"  [ssd] L sample values: {L.flatten()[:5]}")

    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)
    print(f"  [ssd] Y_diag shape: {Y_diag.shape}")
    print(f"  [ssd] Y_diag sample values: {Y_diag.flatten()[:5]}")

    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    print(f"  [ssd] decay_states shape: {decay_states.shape}")
    print(f"  [ssd] decay_states sample values: {decay_states.flatten()[:5]}")

    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)
    print(f"  [ssd] states shape: {states.shape}")
    print(f"  [ssd] states sample values: {states.flatten()[:5]}")

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
        print("  [ssd] Initialized initial_states with zeros.")
        print(f"  [ssd] initial_states sample values: {initial_states.flatten()[:5]}")

    states = torch.cat([initial_states, states], dim=1)
    print(f"  [ssd] states after concatenation shape: {states.shape}")
    print(f"  [ssd] states after concatenation sample values: {states.flatten()[:5]}")

    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    print(f"  [ssd] decay_chunk shape: {decay_chunk.shape}")
    print(f"  [ssd] decay_chunk sample values: {decay_chunk.flatten()[:5]}")

    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    print(f"  [ssd] new_states shape: {new_states.shape}")
    print(f"  [ssd] new_states sample values: {new_states.flatten()[:5]}")

    states, final_state = new_states[:, :-1], new_states[:, -1]
    print(f"  [ssd] states shape after splitting: {states.shape}, final_state shape: {final_state.shape}")
    print(f"  [ssd] states after splitting sample values: {states.flatten()[:5]}")
    print(f"  [ssd] final_state sample values: {final_state.flatten()[:5]}")

    # 4. Compute state -> output conversion per chunk
    state_decay_out = torch.exp(A_cumsum)
    print(f"  [ssd] state_decay_out shape: {state_decay_out.shape}")
    print(f"  [ssd] state_decay_out sample values: {state_decay_out.flatten()[:5]}")

    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    print(f"  [ssd] Y_off shape: {Y_off.shape}")
    print(f"  [ssd] Y_off sample values: {Y_off.flatten()[:5]}")

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    print(f"  [ssd] Y after adding Y_diag and Y_off shape: {Y.shape}")
    print(f"  [ssd] Y after adding Y_diag and Y_off sample values: {Y.flatten()[:5]}")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))
        print(f"[RMSNorm] Initialized RMSNorm with d={d}, eps={eps}")
        print(f"[RMSNorm] weight shape: {self.weight.shape}")
        print(f"[RMSNorm] weight sample values: {self.weight.flatten()[:5]}")

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
            print(f"[RMSNorm] x after gated scaling shape: {x.shape}")
            print(f"[RMSNorm] x after gated scaling sample values: {x.flatten()[:5]}")
        mean_sq = x.pow(2).mean(-1, keepdim=True)
        print(f"[RMSNorm] Mean squared value shape: {mean_sq.shape}")
        print(f"[RMSNorm] mean_sq sample values: {mean_sq.flatten()[:5]}")
        rsqrt = torch.rsqrt(mean_sq + self.eps)
        print(f"[RMSNorm] rsqrt shape: {rsqrt.shape}")
        print(f"[RMSNorm] rsqrt sample values: {rsqrt.flatten()[:5]}")
        x_normalized = x * rsqrt * self.weight
        print(f"[RMSNorm] x_normalized shape: {x_normalized.shape}")
        print(f"[RMSNorm] x_normalized sample values: {x_normalized.flatten()[:5]}")
        return x_normalized


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * torch.sigmoid(x)
