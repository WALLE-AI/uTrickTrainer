from pathlib import Path
import re
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from safetensors.torch import load_file
import re
from tokenizers import Tokenizer

##参考：https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3-moe.ipynb
##MoE中的前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])

        # meta device to reduce memory pressure when initializing the model before loading weights
        meta_device = torch.device("meta")
        self.fc1 = nn.ModuleList([
            nn.Linear(
                cfg["emb_dim"], cfg["moe_intermediate_size"],
                bias=False, dtype=cfg["dtype"], device=meta_device)
            for _ in range(cfg["num_experts"])]
        )
        self.fc2 = nn.ModuleList([
            nn.Linear(
                cfg["emb_dim"], cfg["moe_intermediate_size"],
                bias=False, dtype=cfg["dtype"], device=meta_device
                )
            for _ in range(cfg["num_experts"])]
        )
        self.fc3 = nn.ModuleList([
            nn.Linear(
                cfg["moe_intermediate_size"], cfg["emb_dim"],
                bias=False, dtype=cfg["dtype"], device=meta_device
                )
            for _ in range(cfg["num_experts"])]
        )

    def forward(self, x):
        b, seq_len, embed_dim = x.shape
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)
        
        expert_outputs = []
        for e in range(self.num_experts):
            hidden = torch.nn.functional.silu(self.fc1[e](x)) * self.fc2[e](x)
            out = self.fc3[e](hidden)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(expert_outputs, dim=-2)  # (b, t, num_experts, emb_dim)

        gating_probs = torch.zeros_like(scores)

        for i in range(self.num_experts_per_tok):
            indices = topk_indices[..., i:i+1]
            prob = topk_probs[..., i:i+1]
            gating_probs.scatter_(dim=-1, index=indices, src=prob)
        gating_probs = gating_probs.unsqueeze(-1)  # (b, t, num_experts, 1)
        
        # Weighted sum over experts
        y = (gating_probs * expert_outputs).sum(dim=-2)
        return y


        # For some reason, the version below is slower than the naive version
        # above that computes all experts, even the unused ones

        # def forward(self, x):
        #     scores = self.gate(x)  # (b, seq_len, num_experts)
        #     topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        #     topk_probs = torch.softmax(topk_scores, dim=-1)
        #     y = torch.zeros_like(x)
        #
        #     for i in range(self.num_experts_per_tok):
        #         # expert_indices is (b, seq_len) with values in [0, num_experts)
        #         expert_indices = topk_indices[..., i]
        #         prob = topk_probs[..., i].unsqueeze(-1)  # (b, seq_len, 1)
        #
        #         # For each expert, process only the tokens assigned to it
        #         for e in range(self.num_experts):
        #             mask = (expert_indices == e)  # (b, seq_len) boolean mask
        #             if mask.any():
        #                 selected = x[mask]  # (num_tokens_e, emb_dim)
        #                 out = self.fc3[e](torch.nn.functional.silu(self.fc1[e](selected)) * self.fc2[e](selected))
        #                 y[mask] += prob[mask] * out
        #     return y
        
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
    
    
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


# def apply_rope(x, cos, sin):
#     # x: (batch_size, num_heads, seq_len, head_dim)
#     batch_size, num_heads, seq_len, head_dim = x.shape
#     assert head_dim % 2 == 0, "Head dimension must be even"

#     # Split x into first half and second half
#     x1 = x[..., : head_dim // 2]  # First half
#     x2 = x[..., head_dim // 2 :]  # Second half

#     # Adjust sin and cos shapes
#     cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
#     sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

#     # Apply the rotary transformation
#     rotated = torch.cat((-x2, x1), dim=-1)
#     x_rotated = (x * cos) + (rotated * sin)

#     # It's ok to use lower-precision after applying cos and sin rotation
#     return x_rotated.to(dtype=x.dtype)


def apply_rope(x, cos, sin, pos_offset: int = 0):
    # x: (batch, num_heads, seq_len, head_dim)
    b, h, t, d = x.shape
    assert d % 2 == 0, "Head dimension must be even"

    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    # 关键：切片用 [pos_offset : pos_offset + t]
    cos_slice = cos[pos_offset : pos_offset + t, :].unsqueeze(0).unsqueeze(0)  # (1,1,t,d)
    sin_slice = sin[pos_offset : pos_offset + t, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos_slice) + (rotated * sin_slice)
    return x_rotated.to(dtype=x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, past_kv=None, use_cache: bool=True):
        b, num_tokens, _ = x.shape

        # Proj
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (b,h,t,hd)
        keys    = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2) # (b,g,t,hd)
        values  = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)# (b,g,t,hd)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys    = self.k_norm(keys)

        # === RoPE with absolute position offset ===
        if past_kv is None:
            pos_offset = 0
        else:
            pos_offset = past_kv[0].shape[2]  # t_past

        queries = apply_rope(queries, cos, sin, pos_offset=pos_offset)
        keys    = apply_rope(keys,    cos, sin, pos_offset=pos_offset)

        # Append cache
        if past_kv is not None:
            k_past, v_past = past_kv
            keys   = torch.cat([k_past, keys], dim=2)     # (b,g,t_total,hd)
            values = torch.cat([v_past, values], dim=2)   # (b,g,t_total,hd)

        # Expand to heads
        keys_rep   = keys.repeat_interleave(self.group_size, dim=1)   # (b,h,t_total,hd)
        values_rep = values.repeat_interleave(self.group_size, dim=1) # (b,h,t_total,hd)

        # Attention
        attn_scores = queries @ keys_rep.transpose(2, 3)              # (b,h,t_q,t_total)

        # 更稳的 mask 广播：支持 (t_q,t_kv) 或已带 batch/head 维度
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # -> (1,1,t_q,t_kv)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        context = (attn_weights @ values_rep).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        out = self.out_proj(context)

        if use_cache:
            return out, (keys, values)
        return out


    # def forward(self, x, mask, cos, sin, past_kv=None, use_cache: bool=False):
    #     """
    #     x: (b, t, d)
    #     past_kv: optional tuple (k_cached, v_cached) with shapes:
    #              k_cached: (b, num_kv_groups, t_past, head_dim)
    #              v_cached: (b, num_kv_groups, t_past, head_dim)
    #     use_cache: if True, returns (out, present_kv). Else returns out tensor.
    #     """
    #     b, num_tokens, _ = x.shape

    #     # Proj
    #     queries = self.W_query(x)  # (b, t, h*hd)
    #     keys = self.W_key(x)       # (b, t, g*hd)
    #     values = self.W_value(x)   # (b, t, g*hd)

    #     # Reshape
    #     queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (b,h,t,hd)
    #     keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)    # (b,g,t,hd)
    #     values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)# (b,g,t,hd)

    #     # (optional) norm
    #     if self.q_norm:
    #         queries = self.q_norm(queries)
    #     if self.k_norm:
    #         keys = self.k_norm(keys)

    #     # RoPE
    #     queries = apply_rope(queries, cos, sin)
    #     keys = apply_rope(keys, cos, sin)

    #     # Append cache if provided
    #     if past_kv is not None:
    #         k_past, v_past = past_kv
    #         # concat along time dim
    #         keys = torch.cat([k_past, keys], dim=2)     # (b,g,t_total,hd)
    #         values = torch.cat([v_past, values], dim=2) # (b,g,t_total,hd)

    #     # Repeat K/V to match heads
    #     keys_rep = keys.repeat_interleave(self.group_size, dim=1)     # (b,h,t_total,hd)
    #     values_rep = values.repeat_interleave(self.group_size, dim=1) # (b,h,t_total,hd)

    #     # Attention
    #     attn_scores = queries @ keys_rep.transpose(2, 3)  # (b,h,t,t_total)
    #     if mask is not None:
    #         attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    #     attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
    #     context = (attn_weights @ values_rep).transpose(1, 2).reshape(b, num_tokens, self.d_out)
    #     out = self.out_proj(context)

    #     if use_cache:
    #         # Return KV in grouped form to save memory
    #         return out, (keys, values)
    #     return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        if cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, past_kv=None, use_cache: bool=False):
        # Attention block (with optional cache)
        shortcut = x
        x = self.norm1(x)
        att_out = self.att(x, mask, cos, sin, past_kv=past_kv, use_cache=use_cache)
        if use_cache:
            x, present_kv = att_out
        else:
            x = att_out
            present_kv = None
        x = x + shortcut

        # FFN
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        if use_cache:
            return x, present_kv
        return x

    
class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg


    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
    

QWEN3_CONFIG = {
    "vocab_size": 151_936,
    "context_length": 262_144,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 48,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 10_000_000.0,
    "dtype": torch.bfloat16,
    "num_experts": 128,
    "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
}


def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb


def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        # Q, K, V projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        if "num_experts" in param_config:
            # Load router (gating) weights
            block.ff.gate.weight = assign(
                block.ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight"
            )
            # Load expert weights
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(
                    block.ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight"
                )
                block.ff.fc2[e].weight = assign(
                    block.ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight"
                )
                block.ff.fc3[e].weight = assign(
                    block.ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight"
                )
                # After assigning weights, move the expert layers from meta to CPU
                block.ff.fc1[e] = block.ff.fc1[e].to("cpu")
                block.ff.fc2[e] = block.ff.fc2[e].to("cpu")
                block.ff.fc3[e] = block.ff.fc3[e].to("cpu")

        else:
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )

        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        # Model uses weight tying, hence we reuse the embedding layer weights here
        print("Model uses weight tying.")
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
   
class PipelineQwen2KV(nn.Module):
    """
    Two-GPU pipeline for Qwen3 with KV cache support.

    Layout:
      - cuda:0 -> tok_emb + blocks[:split_at]
      - cuda:1 -> blocks[split_at:] + final_norm + out_head

    Methods:
      - forward(in_idx): no-cache forward (compat)
      - prefill(in_idx): build kv cache for the whole prompt
      - decode(next_token_ids): single-step decode using cache
    """
    def __init__(self, base_model: "Qwen3Model", split_at: int = None,
                 dev0: str = "cuda:0", dev1: str = "cuda:1"):
        super().__init__()
        assert isinstance(base_model, Qwen3Model)
        self.base = base_model
        self.dev0 = torch.device(dev0)
        self.dev1 = torch.device(dev1)

        n_layers = len(self.base.trf_blocks)
        if split_at is None:
            split_at = n_layers // 2
        assert 0 < split_at < n_layers
        self.split_at = split_at

        # Move modules
        self.emb = self.base.tok_emb.to(self.dev0)
        self.blocks0 = nn.ModuleList([self.base.trf_blocks[i] for i in range(0, split_at)]).to(self.dev0)
        self.blocks1 = nn.ModuleList([self.base.trf_blocks[i] for i in range(split_at, n_layers)]).to(self.dev1)
        self.final_norm = self.base.final_norm.to(self.dev1)
        self.head = self.base.out_head.to(self.dev1)

        # RoPE tables：按需搬运
        self.cos0 = self.base.cos.to(self.dev0, non_blocking=True)
        self.sin0 = self.base.sin.to(self.dev0, non_blocking=True)
        self.cos1 = self.base.cos.to(self.dev1, non_blocking=True)
        self.sin1 = self.base.sin.to(self.dev1, non_blocking=True)

        # KV caches: lists per layer slice (grouped shape for memory)
        self.kv0 = [None] * len(self.blocks0)  # each item: (k,v) or None
        self.kv1 = [None] * len(self.blocks1)

        self.seqlen = 0  # tracked total cached length

    def _causal_mask(self, dev, t_q, t_kv):
        """Return causal mask for (t_q, t_kv) attention on device dev. None for single-token decode."""
        # full prefill (t_q == t_kv) -> upper-triangular mask
        if t_q > 1 or t_kv > 1:
            m = torch.triu(torch.ones(t_q, t_kv, device=dev, dtype=torch.bool), diagonal=1 + (t_kv - t_q))
        else:
            # single token attending to all history -> no mask needed
            m = None
        if m is None:
            return None
        # broadcast to (b,h,t_q,t_kv) in attention; we'll use it directly on attn_scores (b,h,*,*)
        # Our attention applies mask already in (b,h,t_q,t_kv) shape. We'll expand in-place there.
        return m  # callers will expand as needed

    @torch.inference_mode()
    def forward(self, in_idx: torch.Tensor):
        """Compatibility path: no-cache forward, returns logits on dev1."""
        bsz, t = in_idx.shape
        x = self.emb(in_idx.to(self.dev0))
        # full-seq mask for each slice
        mask0 = torch.triu(torch.ones(t, t, device=self.dev0, dtype=torch.bool), diagonal=1)
        for blk in self.blocks0:
            x = blk(x, mask0, self.cos0, self.sin0)
        x = x.to(self.dev1, non_blocking=True)
        mask1 = torch.triu(torch.ones(t, t, device=self.dev1, dtype=torch.bool), diagonal=1)
        for blk in self.blocks1:
            x = blk(x, mask1, self.cos1, self.sin1)
        x = self.final_norm(x)
        logits = self.head(x)
        return logits

    @torch.inference_mode()
    def reset_cache(self):
        self.kv0 = [None] * len(self.blocks0)
        self.kv1 = [None] * len(self.blocks1)
        self.seqlen = 0

    @torch.inference_mode()
    def prefill(self, in_idx: torch.Tensor):
        """
        Run the whole prompt once to build KV caches.
        in_idx: (b, t0) LongTensor
        """
        self.reset_cache()
        bsz, t0 = in_idx.shape

        # Stage 0
        x = self.emb(in_idx.to(self.dev0))
        mask0 = torch.triu(torch.ones(t0, t0, device=self.dev0, dtype=torch.bool), diagonal=1)
        for i, blk in enumerate(self.blocks0):
            x, present = blk(x, mask0, self.cos0, self.sin0, past_kv=None, use_cache=True)
            self.kv0[i] = present  # grouped (k,v)

        # Hop to dev1
        x = x.to(self.dev1, non_blocking=True)

        # Stage 1
        mask1 = torch.triu(torch.ones(t0, t0, device=self.dev1, dtype=torch.bool), diagonal=1)
        for j, blk in enumerate(self.blocks1):
            x, present = blk(x, mask1, self.cos1, self.sin1, past_kv=None, use_cache=True)
            self.kv1[j] = present

        self.seqlen = t0
        x = self.final_norm(x)
        logits = self.head(x)  # (b, t0, vocab)
        return logits  # 兼容：prefill 也返回最后一步的logits

    @torch.inference_mode()
    def decode(self, next_token_ids: torch.Tensor):
        """
        Single-step decode with cache.
        next_token_ids: (b, 1) LongTensor
        Returns logits for the new position (b, 1, vocab)
        """
        assert self.seqlen > 0, "Call prefill() before decode()"
        bsz, t1 = next_token_ids.shape
        assert t1 == 1

        # Stage 0
        x = self.emb(next_token_ids.to(self.dev0))
        # Build a mask for (1, seqlen+1) but None is fine for single-step
        mask0 = None
        for i, blk in enumerate(self.blocks0):
            x, present = blk(x, mask0, self.cos0, self.sin0, past_kv=self.kv0[i], use_cache=True)
            self.kv0[i] = present

        # Hop
        x = x.to(self.dev1, non_blocking=True)

        # Stage 1
        mask1 = None
        for j, blk in enumerate(self.blocks1):
            x, present = blk(x, mask1, self.cos1, self.sin1, past_kv=self.kv1[j], use_cache=True)
            self.kv1[j] = present

        self.seqlen += 1
        x = self.final_norm(x)
        logits = self.head(x)  # (b, 1, vocab)
        return logits


# def build_pipeline_model(base_model: "Qwen3Model",
#                          devices: list[str] | None = None) -> nn.Module:
#     """
#     Wrap Qwen3Model into a pipeline-parallel model when there are >=2 CUDA devices.
#     Fallbacks to the original single-GPU model if only one GPU is available.

#     Usage:
#         model = build_pipeline_model(model, devices=["cuda:0","cuda:1","cuda:2","cuda:3"])
#     """
#     if devices is None:
#         n = torch.cuda.device_count()
#         if n <= 1:
#             # single-GPU fallback: just move to cuda:0 if available
#             if n == 1:
#                 return base_model.to("cuda:0")
#             return base_model
#         devices = [f"cuda:{i}" for i in range(n)]
#     if len(devices) == 1:
#         return base_model.to(devices[0])
#     return PipelineQwen2KV(base_model, devices)  

def build_2gpu_kv(model: "Qwen3Model", split_at: int = None):
    return PipelineQwen2KV(model, split_at=split_at or (len(model.trf_blocks)//2),
                           dev0="cuda:0", dev1="cuda:1")


   
repo_id_dir = "/home/dataset0/images/Qwen3-30B-A3B-Thinking-2507"  # New thinking model
def qwen3_main():      

    index_path = os.path.join(repo_id_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in set(index["weight_map"].values()):
        shard_path = os.path.join(repo_id_dir, filename)
        shard = load_file(shard_path)
        weights_dict.update(shard)
    dtype=torch.bfloat16
    model = Qwen3Model(QWEN3_CONFIG)
    # model = model.to("meta")  # Initialize on meta device to reduce memory pressure
    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    model = build_2gpu_kv(model)
    print(f"Model loaded with {model_memory_size(model):.2f} GB memory footprint.")
    return model




class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None,
                apply_chat_template=True, add_generation_prompt=False, add_thinking=False):

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token_id = self._special_to_id.get("<|endoftext|>")
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    # def _wrap_chat(self, user_msg):
    #     s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
    #     if self.add_generation_prompt:
    #         s += "<|im_start|>assistant"
    #         if self.add_thinking:
    #             s += "\n"
    #         else:
    #             s += "\n<think>\n\n</think>\n\n"
    #     return s
    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n<think>\n"  # 只打开，不提前闭合
            else:
                s += "\n"
        return s


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                   and torch.all(next_token == eos_token_id)):
               break

            yield next_token
            
            token_ids = torch.cat([token_ids, next_token], dim=1)
            
            
def qwen3_generate(model,input_token_ids,tokenizer):
            
    input_token_ids_tensor = torch.tensor(input_token_ids,device="cuda").unsqueeze(0)
    for token in generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=500,
        # eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )
        
@torch.inference_mode()    
def qwen3_generate_kv(model_pp: "PipelineQwen2KV", input_token_ids, tokenizer, max_new_tokens=200):
    # 组 batch=1
    input_token_ids_tensor = torch.tensor(input_token_ids,device="cuda").unsqueeze(0)
    # 1) Prefill 阶段：把整段 prompt 过一遍，建立 KV 缓存
    _ = model_pp.prefill(input_token_ids_tensor)
    last_token = input_token_ids_tensor[:, -1:]

    # 2) 单步解码
    for _ in range(max_new_tokens):
        logits = model_pp.decode(last_token)          # (1,1,vocab)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)

        # 打印增量
        piece = tokenizer.decode(next_token.squeeze(0).tolist())
        print(piece, end="", flush=True)

        # 终止判断
        if tokenizer.eos_token_id is not None and torch.all(next_token == tokenizer.eos_token_id):
            break

        # 准备下一个 token
        last_token = next_token.cpu()
        
        
@torch.inference_mode()
def generate_text_stream_kv(model_pp: "PipelineQwen2KV",
                            input_ids: torch.Tensor,
                            tokenizer: "Qwen3Tokenizer",
                            max_new_tokens: int = 1024,
                            eos_token_id: int | None = None):
    """
    model_pp: PipelineQwen2KV
    input_ids: (t0,) or (b,t0) LongTensor
    """
    # if input_ids.dim() == 1:
    #     input_ids = input_ids.unsqueeze(0)
    # Prefill
    _ = model_pp.prefill(input_ids)

    for _ in range(max_new_tokens):
        # greedy
        logits = model_pp.decode(torch.tensor([[tokenizer.eos_token_id if False else 0]], device="cpu"))  # dummy; will replace below
        # 直接用上一步最后一个logits更合理，这里改为对 decode 输入 last token
        # 我们需要最后一步的已生成 token，prefill 之后没有新 token，这里用 input_ids[:,-1]
        last_tok = input_ids[:, -1:]
        logits = model_pp.decode(last_tok)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (b,1)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        # 输出字片段
        print(tokenizer.decode(next_token[0].tolist()), end="", flush=True)

        # 滚动输入（仅用于下一次decode的入参）
        input_ids = torch.cat([input_ids, next_token.cpu()], dim=1)


def qwen3_tokenizer_main():
    tokenizer_file_path = repo_id_dir+"/tokenizer.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id_dir,
        add_generation_prompt=True,
        add_thinking=True
    )

    # prompt = "Give me a short introduction to large language models."
    prompt = "说一下居家智能体把"
    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)
    return input_token_ids,tokenizer


import math
import torch

def sample_next_token_from_logits(
    logits: torch.Tensor,              # (B, V) 或 (V,)
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    返回采样后的 token 索引，形状 (B,1) 或 (1,)（若输入是 (V,)）
    约定：先做温度，再 top-k，再 nucleus(top-p)
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # (1, V)
        squeeze_back = True
    else:
        squeeze_back = False

    # 温度
    if temperature is not None and temperature > 0 and not math.isclose(temperature, 1.0):
        logits = logits / temperature

    # Top-k
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float('-inf')), logits)

    # Top-p (nucleus)
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        mask = cdf > top_p
        # 保证至少留下第一个
        mask[..., 0] = False
        # 将被过滤的置 -inf
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        # 还原到原索引位置
        logits_zero = torch.full_like(logits, float('-inf'))
        logits = logits_zero.scatter(-1, sorted_idx, sorted_logits)

    # 归一化并采样
    probs = torch.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)  # (B,1)

    if squeeze_back:
        next_tok = next_tok.squeeze(0)  # (1,)
    return next_tok


def pad_to_same_length(list_of_token_ids: list[list[int]], pad_id: int) -> torch.Tensor:
    """
    右侧填充到相同长度。返回 LongTensor(B, T).
    """
    max_len = max(len(x) for x in list_of_token_ids)
    out = []
    for x in list_of_token_ids:
        if len(x) < max_len:
            x = x + [pad_id] * (max_len - len(x))
        out.append(x)
    return torch.tensor(out, dtype=torch.long)

@torch.inference_mode()
def batched_generate_kv_sample(
    model_pp: "PipelineQwen2KV",
    batch_input_ids: list[list[int]],  # 每条提示的 ids（长度尽量一致；否则会右侧 pad）
    tokenizer: "Qwen3Tokenizer",
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    seed: int | None = None,
    stream_callback=None,             # 可选：func(step, seq_idx, text_piece) -> None
):
    """
    - 先 prefill()（构建共享 KV），再逐步 decode()。
    - 采样策略：温度/Top-k/Top-p，支持组合。
    - 简易“流式回调”：每步对每条序列输出新增 token 对应的文本片段。
    """
    if seed is not None:
        torch.manual_seed(seed)

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # 右侧 pad（无 padding mask，pad 也会被当作普通 token 参与 prefill；建议用 eos 作为 pad）
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    assert pad_id is not None, "Tokenizer 需要有 eos 或 pad token 用于等长填充"
    input_ids = pad_to_same_length(batch_input_ids, pad_id)  # (B, T0)

    B, T0 = input_ids.shape
    # 1) Prefill：把整段 prompt 过一遍
    _ = model_pp.prefill(input_ids)

    # 每条序列的“是否完成”标记
    finished = torch.zeros(B, dtype=torch.bool)
    # 每条序列“上一 token”（初始为各自最后一个 token）
    last_tokens = input_ids[:, -1:].clone()  # (B,1)

    # 已生成的 token 计数（不含 prompt）
    for step in range(max_new_tokens):
        # 2) 单步 decode：得到 (B,1,V) logits
        logits = model_pp.decode(last_tokens)       # (B,1,V)
        step_logits = logits[:, -1, :]             # (B,V)

        # 对“已完成”的序列，把 logits 限制为只会采样到 eos
        if eos_token_id is not None:
            done_mask = finished.unsqueeze(1)  # (B,1)
            if done_mask.any():
                step_logits = torch.where(
                    done_mask,
                    torch.full_like(step_logits, float('-inf')).scatter(1, torch.full((done_mask.sum(),1), eos_token_id, device=step_logits.device, dtype=torch.long), 0.0),
                    step_logits
                )

        # 3) 采样
        next_tokens = sample_next_token_from_logits(
            step_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )  # (B,1)

        # 流式输出（逐条）
        if stream_callback is None:
            # 默认直接 print
            for bi in range(B):
                if not finished[bi]:
                    piece = tokenizer.decode(next_tokens[bi].tolist())
                    print(piece, end="", flush=True)
            # 可选：打印换行（每步不要换行，直到结束）
        else:
            for bi in range(B):
                if not finished[bi]:
                    piece = tokenizer.decode(next_tokens[bi].tolist())
                    stream_callback(step, bi, piece)

        # 4) 终止判断
        if eos_token_id is not None:
            finished = finished | (next_tokens.squeeze(1) == eos_token_id)

        # 所有序列都结束则停止
        if torch.all(finished):
            break

        # 5) 准备下一步
        last_tokens = next_tokens.cpu()  # decode() 内部会把它搬到 cuda:0

    # 最后统一换行（如果用默认 print）
    if stream_callback is None:
        print()




def qwen3_generate_kv_sample_single(
    model_pp: "PipelineQwen2KV",
    input_token_ids: list[int],
    tokenizer: "Qwen3Tokenizer",
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
):
    _ = model_pp.prefill(torch.tensor(input_token_ids).unsqueeze(0))
    last = torch.tensor(input_token_ids[-1:]).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model_pp.decode(last)          # (1,1,V)
        next_tok = sample_next_token_from_logits(
            logits[:, -1, :],
            temperature=temperature, top_k=top_k, top_p=top_p
        )  # (1,1)
        piece = tokenizer.decode(next_tok.squeeze(0).tolist())
        print(piece, end="", flush=True)
        if next_tok.item() == tokenizer.eos_token_id:
            break
        last = next_tok.cpu()
    print()


if __name__ == "__main__":
    input_token_ids,tokenizer = qwen3_tokenizer_main()
    model = qwen3_main()
    qwen3_generate_kv(model,input_token_ids,tokenizer,max_new_tokens=1024)


            
            
