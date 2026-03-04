# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.models.common.attention import (
    AttentionMasksType,
    VarlenMetadata,
    create_attention_mask,
    get_causal_mask_mod,
)

logger = logging.getLogger(__name__)


class NoOpDataLoader(BaseDataLoader):
    """A no-op dataloader for use when verl manages its own data loading.

    Satisfies the BaseDataLoader interface required by torchtitan's Trainer
    but does nothing. Its __iter__ yields nothing, and state_dict /
    load_state_dict are no-ops.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        pass

    def __init__(self, **kwargs):
        pass

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        return iter([])

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# Mapping from HuggingFace model_type to torchtitan model name.
# Torchtitan models not mapped here:
#   - flux: diffusion model, not applicable to verl's RL/SFT workflows
#   - llama3_ft: fault-tolerant variant of llama3, same HF models (mapped via "llama")
_HF_MODEL_TYPE_TO_TORCHTITAN_NAME = {
    "qwen2": "qwen3",
    "qwen3": "qwen3",
    "qwen2_moe": "qwen3",
    "qwen3_moe": "qwen3",
    "llama": "llama3",
    "llama4": "llama4",
    "deepseek_v3": "deepseek_v3",
    "gpt_oss": "gpt_oss",
}


def derive_torchtitan_name_and_flavor(hf_config) -> tuple[str, str]:
    """Derive torchtitan model name and flavor from a HuggingFace config.

    The name is mapped from ``hf_config.model_type``. The flavor is found by
    matching architecture parameters (dim, n_layers, vocab_size) against the
    known flavors registered in the torchtitan model package.

    Args:
        hf_config: A HuggingFace AutoConfig object.

    Returns:
        A ``(name, flavor)`` tuple.

    Raises:
        ValueError: If model_type is unsupported or no matching flavor is found.
    """
    model_type = getattr(hf_config, "model_type", None)
    if model_type is None:
        raise ValueError("HuggingFace config does not have 'model_type' field")

    name = _HF_MODEL_TYPE_TO_TORCHTITAN_NAME.get(model_type)
    if name is None:
        raise ValueError(
            f"Cannot derive torchtitan model name from HF model_type '{model_type}'. "
            f"Supported types: {list(_HF_MODEL_TYPE_TO_TORCHTITAN_NAME.keys())}."
        )

    # Import the model package and find the configs dict
    model_module = importlib.import_module(f"torchtitan.models.{name}")
    model_configs = None
    for attr in dir(model_module):
        obj = getattr(model_module, attr)
        if isinstance(obj, dict) and attr.endswith("_configs"):
            model_configs = obj
            break

    if model_configs is None:
        raise ValueError(
            f"Could not find model configs dict in torchtitan.models.{name}. "
            f"Expected a dict attribute ending with '_configs'."
        )

    hidden_size = hf_config.hidden_size
    num_layers = hf_config.num_hidden_layers
    vocab_size = hf_config.vocab_size

    for flavor_name, model_cfg in model_configs.items():
        if (
            getattr(model_cfg, "dim", None) == hidden_size
            and getattr(model_cfg, "n_layers", None) == num_layers
            and getattr(model_cfg, "vocab_size", None) == vocab_size
        ):
            logger.info(
                f"Auto-derived torchtitan name='{name}', flavor='{flavor_name}' from HF model_type='{model_type}'"
            )
            return name, flavor_name

    raise ValueError(
        f"No matching torchtitan flavor found for model_type='{model_type}' "
        f"(hidden_size={hidden_size}, num_hidden_layers={num_layers}, "
        f"vocab_size={vocab_size}). "
        f"Available flavors for '{name}': {list(model_configs.keys())}."
    )


def enable_fsdp_gradient_division(model: nn.Module, dp_size: int) -> None:
    """
    Re-enable FSDP's automatic gradient division.

    TorchTitan calls disable_fsdp_gradient_division() which sets gradient_divide_factor=1.0.
    This re-enables it by setting the factor to the specified dp_size, so gradients are
    averaged across FSDP ranks. This is needed for verl's loss scaling (loss * dp_size)
    to work correctly.

    Args:
        model: The model (or model part) to enable gradient division on.
        dp_size: The data parallel size to use as the gradient divide factor.
    """

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(float(dp_size))


def get_attention_masks(
    input_batch: torch.Tensor,
    positions: torch.Tensor,
    attn_type: str,
) -> AttentionMasksType:
    match attn_type:
        case "flex":
            return _get_flex_attention_masks(
                input_batch,
                positions,
            )
        case "varlen":
            return _create_varlen_metadata_for_document(
                input_batch,
                positions,
            )
        case _:
            raise TypeError("Only varlen and flex attn masks are supported")


def _get_document_mask_mod(positions: torch.Tensor) -> _mask_mod_signature:
    # Detect boundaries from position resets
    first_dummy_value = positions[:, :1] - 1
    position_diff = torch.diff(positions, prepend=first_dummy_value, dim=-1)
    sequence_indices = (position_diff != 1).cumsum(-1)  # [batch, seq]

    def document_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


def _get_flex_attention_masks(
    input_batch: torch.Tensor,
    positions: torch.Tensor,
) -> AttentionMasksType:
    mask_mods = [get_causal_mask_mod()]
    B = input_batch.shape[0]
    mask_mods.append(_get_document_mask_mod(positions=positions))
    return create_attention_mask(and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1])


def _create_varlen_metadata_for_document(input_batch: torch.Tensor, positions: torch.Tensor) -> VarlenMetadata:
    """
    Creates cumulative sequence length indices needed for variable length attention

    Args:
        input_batch: Input token IDs with shape [batch, seq].
        positions: Position IDs with shape [batch, seq]. Boundaries detected where
            position diff != 1 (i.e., position resets).

    Returns:
        VarlenMetadata containing cumulative sequence length indices for q, k, and max_seq_len
    """
    batch_size, seq_len = input_batch.shape
    device = input_batch.device

    # Detect boundaries from position resets (where diff != 1)
    first_dummy_value = positions[:, :1] - 1
    position_diff = torch.diff(positions, prepend=first_dummy_value, dim=-1)
    # boundary_mask[b, i] is True if position i starts a new document
    boundary_mask = position_diff != 1  # [batch, seq]
    boundary_mask[:, 0] = True

    cu_seqlens_list, all_seq_lengths = [], []
    offset = 0

    for b in range(batch_size):
        # Find positions where new documents start
        boundary_positions = boundary_mask[b].nonzero(as_tuple=True)[0].to(torch.int32)
        sample_cu_seqlens = torch.cat(
            [
                boundary_positions,
                torch.tensor([seq_len], dtype=torch.int32, device=device),
            ]
        )
        sample_cu_seqlens = torch.unique_consecutive(sample_cu_seqlens)

        seq_lengths = torch.diff(sample_cu_seqlens)
        all_seq_lengths.append(seq_lengths)

        cu_seqlens_adjusted = sample_cu_seqlens[:-1] + offset
        cu_seqlens_list.append(cu_seqlens_adjusted)

        offset += seq_len

    packed_cu_seqlens = torch.cat(cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)])

    max_seqlen = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths = torch.cat(all_seq_lengths)
        # device to host sync but only done once per model forward
        max_seqlen = all_seq_lengths.max().item()

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )


def iter_per_tensor_params_ep(
    params: dict[str, Any],
    device: int,
    ep_group: torch.distributed.ProcessGroup,
    ep_size: int,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield (name, tensor) pairs for weight sync with Expert Parallel.

    Gathers expert weights across EP ranks one (layer, weight_type) group
    at a time to avoid OOM from materializing all experts simultaneously.

    Non-expert params are yielded first (with FSDP full_tensor() if needed),
    then expert params are all-gathered per group and yielded individually.

    Args:
        params: HF-format state dict with per-expert keys. Expert keys must
            follow the pattern ``model.layers.{L}.mlp.experts.{E}.{suffix}``.
        device: CUDA device ID to place tensors on.
        ep_group: The EP process group for all-gather.
        ep_size: Number of EP ranks.
    """
    # Separate expert and non-expert params.
    # Group expert params by (layer_id, weight_type) for batched all-gather.
    # (layer_id, weight_suffix) -> {expert_id: (name, param)}
    expert_params: dict[tuple[int, str], dict[int, tuple[str, Any]]] = {}
    non_expert_params: list[tuple[str, Any]] = []

    for name, param in params.items():
        if "mlp.experts." not in name:
            non_expert_params.append((name, param))
            continue

        # Parse: model.layers.{L}.mlp.experts.{E}.{weight_type}
        parts = name.split(".")
        layer_id = None
        expert_id = None
        weight_suffix = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_id = int(parts[i + 1])
                except ValueError as err:
                    raise ValueError(
                        f"Expected integer layer ID after 'layers.' in '{name}', got '{parts[i + 1]}'"
                    ) from err
            elif part == "experts" and i + 1 < len(parts):
                try:
                    expert_id = int(parts[i + 1])
                except ValueError as err:
                    raise ValueError(
                        f"Expected integer expert ID after 'experts.' in '{name}', got '{parts[i + 1]}'"
                    ) from err
                weight_suffix = ".".join(parts[i + 2 :])

        if layer_id is None or expert_id is None or weight_suffix is None:
            raise ValueError(
                f"Failed to parse expert param name '{name}'. "
                f"Expected format: 'model.layers.{{L}}.mlp.experts.{{E}}.{{weight_type}}' "
                f"where L and E are integers. "
                f"Parsed: layer_id={layer_id}, expert_id={expert_id}, weight_suffix={weight_suffix}"
            )
        key = (layer_id, weight_suffix)
        if key not in expert_params:
            expert_params[key] = {}
        expert_params[key][expert_id] = (name, param)

    # Release original params dict - all refs are now in expert_params/non_expert_params
    params.clear()

    # Yield non-expert params first, then release them
    for name, param in non_expert_params:
        if isinstance(param, DTensor):
            yield name, param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
        else:
            yield name, param
    del non_expert_params

    # Yield expert params: all-gather across EP ranks, one (layer, weight_type) at a time
    for (layer_id, weight_suffix), experts_dict in sorted(expert_params.items()):
        # Stack local expert weights into a single tensor for all-gather
        sorted_expert_ids = sorted(experts_dict.keys())
        local_weights = []
        for eid in sorted_expert_ids:
            _, param = experts_dict[eid]
            # DTensor case: EP+FSDP — need full_tensor() to resolve FSDP sharding.
            if isinstance(param, DTensor):
                param = param.to(device, non_blocking=True).full_tensor()
            # Plain Tensor case: EP only — just move to GPU.
            else:
                assert isinstance(param, torch.Tensor), (
                    f"Expected DTensor or Tensor for expert param '{experts_dict[eid][0]}', got {type(param)}"
                )
                param = param.to(device, non_blocking=True)
            local_weights.append(param)

        # Build the name template before releasing experts_dict refs
        name_template = experts_dict[sorted_expert_ids[0]][0]
        parts = name_template.split(".")
        for i, part in enumerate(parts):
            if part == "experts" and i + 1 < len(parts):
                parts[i + 1] = "{}"
                break
        name_template = ".".join(parts)

        # Stack into [num_local_experts, *weight_shape]
        local_stacked = torch.stack(local_weights, dim=0)

        # All-gather across EP ranks: each rank contributes num_local_experts
        gathered_list = [torch.empty_like(local_stacked) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered_list, local_stacked, group=ep_group)

        # Concatenate: [total_num_experts, *weight_shape]
        all_experts = torch.cat(gathered_list, dim=0)
        num_total_experts = all_experts.shape[0]

        for expert_id in range(num_total_experts):
            expert_name = name_template.format(expert_id)
            # .clone() creates an independent tensor so del all_experts actually frees memory
            yield expert_name, all_experts[expert_id].to(torch.bfloat16).clone()

        # Free memory - safe now since all yielded tensors are clones, not views
        del local_weights, local_stacked, gathered_list, all_experts
        torch.cuda.empty_cache()
