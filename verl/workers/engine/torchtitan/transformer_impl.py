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
"""
The concrete Engine implementation using PyTorch TorchTitan parallelism (FSDP2 + TP + PP)
"""

import gc
import importlib
import logging
import os
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.train import Trainer

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.torchtitan.utils import (
    NoOpDataLoader,
    derive_torchtitan_name_and_flavor,
    enable_fsdp_gradient_division,
    get_attention_masks,
    pad_microbatch_to_length,
)

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches

IGNORE_INDEX = -100

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class TorchTitanEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch TorchTitan parallelism.

    Supports model sharding with FSDP2, tensor parallelism, activation/optimizer offloading,
    LoRA, and sequence parallelism following the TorchTitan design.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: TorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the TorchTitanEngine.

        Sets up distributed device meshes for tensor and data parallelism, LoRA, and offload policies.

        Args:
            model_config: Configuration for HuggingFace model.
            engine_config: Configuration for FSDP/TorchTitan engine (uses FSDP2).
            optimizer_config: Configuration for optimizer.
            checkpoint_config: Configuration for checkpointing.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        # Derive torchtitan model name and flavor from HF config
        torchtitan_name, torchtitan_flavor = derive_torchtitan_name_and_flavor(self.model_config.hf_config)

        # Get ModelSpec from model registry
        model_module = importlib.import_module(f"torchtitan.models.{torchtitan_name}")
        model_spec = model_module.model_registry(torchtitan_flavor)

        # Override attn_backend on the model config if needed
        attn_type = self.engine_config.attn_type
        if hasattr(model_spec.model, "layer") and hasattr(model_spec.model.layer, "attention"):
            model_spec.model.layer.attention.attn_backend = attn_type

        optimizer = OptimizersContainer.Config(
            name=self.optimizer_config.name,
            lr=self.optimizer_config.lr,
            eps=self.optimizer_config.eps,
            beta1=self.optimizer_config.betas[0],
            beta2=self.optimizer_config.betas[1],
            weight_decay=self.optimizer_config.weight_decay,
        )

        total_steps = self.optimizer_config.total_training_steps
        lr_warmup_steps = self.optimizer_config.lr_warmup_steps
        if lr_warmup_steps is None or lr_warmup_steps <= 0:
            lr_warmup_steps = int(self.optimizer_config.lr_warmup_steps_ratio * total_steps)

        lr_scheduler = LRSchedulersContainer.Config(
            warmup_steps=lr_warmup_steps,
            decay_type=self.optimizer_config.decay_type,
            min_lr_factor=self.optimizer_config.min_lr_factor,
        )
        parallelism = ParallelismConfig(
            data_parallel_replicate_degree=self.engine_config.data_parallel_replicate_size,
            data_parallel_shard_degree=self.engine_config.data_parallel_shard_size,
            fsdp_reshard_after_forward=self.engine_config.reshard_after_forward,
            tensor_parallel_degree=self.engine_config.tensor_parallel_size,
            pipeline_parallel_degree=self.engine_config.pipeline_parallel_size,
            context_parallel_degree=self.engine_config.context_parallel_size,
            expert_parallel_degree=self.engine_config.expert_parallel_size,
            expert_tensor_parallel_degree=self.engine_config.expert_tensor_parallel_size,
        )
        checkpoint = CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            initial_load_path=model_config.path,
        )
        compile_config = CompileConfig(enable=self.engine_config.use_torch_compile)
        training_kwargs = {}
        if self.engine_config.max_seq_len is not None:
            training_kwargs["seq_len"] = self.engine_config.max_seq_len
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            training = TrainingConfig(enable_cpu_offload=True, **training_kwargs)
        else:
            training = TrainingConfig(**training_kwargs)

        # Construct Torchtitan's Trainer.Config
        self.config = Trainer.Config(
            model_spec=model_spec,
            hf_assets_path=self.model_config.path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallelism=parallelism,
            checkpoint=checkpoint,
            compile=compile_config,
            training=training,
            # Use a no-op dataloader since verl has its own data loading
            dataloader=NoOpDataLoader.Config(),
        )
        self.trainer = Trainer(self.config)

        self._init_device_mesh()

        # Re-enable FSDP's gradient division for verl's loss scaling.
        # TorchTitan disables gradient division by default (for global token normalization),
        # but verl's loss function multiplies by dp_size to compensate for gradient averaging.
        if self.engine_config.data_parallel_shard_size > 1:
            dp_size = self.get_data_parallel_size()
            for model_part in self.trainer.model_parts:
                enable_fsdp_gradient_division(model_part, dp_size)

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        is_collect = True
        # TP: outputs are on TP rank 0
        if self.parallel_dims.tp > 1:
            tp_mesh = self.parallel_dims.get_optional_mesh("tp")
            is_collect = is_collect and (tp_mesh.get_local_rank() == 0)
        # PP: outputs are on the last PP rank
        if self.parallel_dims.pp > 1:
            pp_mesh = self.parallel_dims.get_optional_mesh("pp")
            is_collect = is_collect and (pp_mesh.get_local_rank() == self.parallel_dims.pp - 1)
        # CP: outputs are on CP rank 0
        if self.parallel_dims.cp > 1:
            cp_mesh = self.parallel_dims.get_optional_mesh("cp")
            is_collect = is_collect and (cp_mesh.get_local_rank() == 0)
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler with TorchTitan parallelism.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager.
        """
        self.module = self.trainer.model_parts
        self.checkpointer = self.trainer.checkpointer
        # load initial HF weights
        self.checkpointer.load()

        if not self.engine_config.forward_only:
            self.optimizer = self.trainer.optimizers
            self.lr_scheduler = self.trainer.lr_schedulers
        else:
            self.optimizer = None
            self.lr_scheduler = None

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        """Initialize the device mesh for TorchTitan style parallelism."""
        world_size = torch.distributed.get_world_size()
        self.parallel_dims = ParallelDims(
            dp_shard=self.engine_config.data_parallel_shard_size,
            dp_replicate=self.engine_config.data_parallel_replicate_size,
            cp=self.engine_config.context_parallel_size,
            tp=self.engine_config.tensor_parallel_size,
            pp=self.engine_config.pipeline_parallel_size,
            ep=self.engine_config.expert_parallel_size,
            etp=self.engine_config.expert_tensor_parallel_size,
            world_size=world_size,
        )
        self.device_mesh = self.parallel_dims.build_mesh()

    def train_mode(self, **kwargs):
        """Return a context manager for training mode."""
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """Return a context manager for evaluation mode."""
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        mesh = self._get_data_parallel_mesh()
        return 0 if mesh is None else mesh.get_local_rank()

    def get_data_parallel_size(self):
        return self.engine_config.data_parallel_shard_size * self.engine_config.data_parallel_replicate_size

    def get_data_parallel_group(self):
        mesh = self._get_data_parallel_mesh()
        if mesh is not None:
            return mesh.get_group()
        # If world_size == dp_size (e.g. single GPU, or all ranks are DP),
        # return WORLD so that collective ops in _postprocess_output
        # (allgather_dict_into_dict, all_reduce) still run and produce the
        # correct metric aggregation format.
        if torch.distributed.get_world_size() == self.get_data_parallel_size():
            return torch.distributed.group.WORLD
        return None

    def _get_data_parallel_mesh(self):
        """Get the data parallel mesh, handling hybrid/fully/replicate shard modes."""
        mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        if mesh is None:
            mesh = self.parallel_dims.get_optional_mesh("fsdp")
        if mesh is None:
            mesh = self.parallel_dims.get_optional_mesh("dp_replicate")
        return mesh

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """Perform forward and optionally backward pass on a batch."""
        tu.assign_non_tensor(data, sp_size=self.engine_config.tensor_parallel_size)

        # Compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        dp_group = self.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        if self.parallel_dims.pp_enabled:
            return self._forward_backward_batch_pp(data, loss_function, forward_only)
        else:
            return self._forward_backward_batch_no_pp(data, loss_function, forward_only)

    def _forward_backward_batch_no_pp(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """Non-PP path: loop over micro-batches with individual forward/backward."""
        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                loss, output = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    loss.backward()
            output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def _forward_backward_batch_pp(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """PP path: pad micro-batches to uniform length and use pp_schedule.step()."""
        # Step 1: Create micro-batches (at least pp_degree for pipeline filling)
        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
            min_num_micro_batch=self.parallel_dims.pp,
        )
        n_microbatches = len(micro_batches)

        # Step 2: Find max sequence length across all micro-batches and sync across DP
        local_max_seq_len = max(mb["input_ids"].shape[-1] for mb in micro_batches)
        max_seq_len_tensor = torch.tensor([local_max_seq_len], device=get_device_id())
        dp_group = self.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(max_seq_len_tensor, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        max_seq_len = int(max_seq_len_tensor.item())

        # Step 3: Pad all micro-batches to max_seq_len and stack
        pad_token_id = tu.get_non_tensor_data(data=data, key="pad_token_id", default=0)
        padded_input_ids_list = []
        padded_position_ids_list = []
        padded_attention_mask_list = []

        for mb in micro_batches:
            mb = mb.to(get_device_id())
            input_ids, position_ids, attention_mask = pad_microbatch_to_length(
                mb, max_seq_len, pad_token_id=pad_token_id
            )
            padded_input_ids_list.append(input_ids)
            padded_position_ids_list.append(position_ids)
            padded_attention_mask_list.append(attention_mask)

        # Shape: [n_microbatches, max_seq_len]
        global_input_ids = torch.cat(padded_input_ids_list, dim=0)
        global_position_ids = torch.cat(padded_position_ids_list, dim=0)
        # global_attention_mask = torch.cat(padded_attention_mask_list, dim=0)

        # Step 4: Build labels (shifted input_ids)
        global_labels = torch.roll(global_input_ids, shifts=-1, dims=1)

        # Step 5: Build attention masks in the format torchtitan expects
        attn_type = self.trainer.model_config.layer.attention.attn_backend
        attention_masks = get_attention_masks(
            input_batch=global_input_ids,
            positions=global_position_ids,
            attn_type=attn_type,
        )

        # Step 6: Handle context parallel if enabled
        extra_kwargs: dict[str, Any] = {"attention_masks": attention_masks}
        if self.parallel_dims.cp_enabled:
            global_input_ids, global_labels, extra_kwargs = prepare_context_parallel_input(
                global_input_ids,
                global_labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.trainer.device,
                self.trainer.config.parallelism.context_parallel_load_balancer,
            )

        # Step 7: Set up the wrapper loss function and run pp_schedule.step()
        pp_schedule = self.trainer.pp_schedule
        pp_has_first_stage = self.trainer.pp_has_first_stage
        pp_has_last_stage = self.trainer.pp_has_last_stage

        # Update n_microbatches on the schedule to match our batch
        pp_schedule._n_microbatches = n_microbatches

        if forward_only and loss_function is None:
            # Forward-only without loss: use return_outputs=True to get logits
            pp_schedule._loss_fn = None
            pp_schedule._has_backward = False

            with torch.no_grad():
                with self.trainer.train_context():
                    if pp_has_first_stage:
                        merged_logits = pp_schedule.step(
                            global_input_ids,
                            positions=global_position_ids,
                            attention_masks=extra_kwargs["attention_masks"],
                            target=None,
                            losses=None,
                            return_outputs=True,
                        )
                    else:
                        merged_logits = pp_schedule.step(
                            attention_masks=extra_kwargs["attention_masks"],
                            target=None,
                            losses=None,
                            return_outputs=True,
                        )

            if pp_has_last_stage and merged_logits is not None:
                # Process merged logits back into per-microbatch model outputs
                # merged_logits: [n_microbatches, max_seq_len, vocab_size]
                micro_batches_on_device = [mb.to(get_device_id()) for mb in micro_batches]
                logits_chunks = torch.tensor_split(merged_logits, n_microbatches, dim=0)
                labels_chunks = torch.tensor_split(global_labels, n_microbatches, dim=0)

                output_lst = []
                for i, (logits_mb, labels_mb) in enumerate(zip(logits_chunks, labels_chunks, strict=False)):
                    mb_data = micro_batches_on_device[i]
                    cu_seqlens = mb_data["input_ids"].offsets()
                    temperature = mb_data["temperature"]
                    calculate_entropy = tu.get_non_tensor_data(data=mb_data, key="calculate_entropy", default=False)
                    seq_lengths = cu_seqlens.diff()

                    # Remove padding
                    valid_logits = []
                    valid_labels = []
                    for j, length in enumerate(seq_lengths):
                        valid_logits.append(logits_mb[j, :length])
                        valid_labels.append(labels_mb[j, :length])
                    logits_rmpad = torch.cat(valid_logits, dim=0)
                    labels_rmpad = torch.cat(valid_labels, dim=0)

                    logits_rmpad = logits_rmpad / temperature
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=labels_rmpad, inplace_backward=True)
                    log_probs_nested = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)

                    model_output = {"log_probs": log_probs_nested}
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        model_output["entropy"] = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

                    output_lst.append(
                        {
                            "model_output": model_output,
                            "loss": 0.0,
                            "metrics": {},
                        }
                    )
                return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
            else:
                return {
                    "model_output": {},
                    "loss": [-1.0],
                    "metrics": {},
                }
        else:
            # Training or forward-only with loss: use wrapper loss_fn
            # The wrapper converts logits -> log_probs/entropy -> verl loss
            # and captures metrics as a side-effect.
            captured_metrics = []
            captured_model_outputs = []
            mb_counter = [0]

            # Pre-split the micro-batch data for the wrapper to access
            micro_batches_on_device = [mb.to(get_device_id()) for mb in micro_batches]

            def _pp_wrapper_loss_fn(pred, labels):
                """Wrapper that adapts verl's loss_function to PP schedule's loss_fn(pred, labels) signature.

                This is called once per micro-batch by the PP schedule, only on the last stage.
                pred: [microbatch_size, seq_len, vocab_size] -- raw logits
                labels: [microbatch_size, seq_len] -- shifted input_ids
                """
                mb_idx = mb_counter[0]
                mb_counter[0] += 1
                mb_data = micro_batches_on_device[mb_idx]

                # Get per-microbatch cu_seqlens from the nested input_ids
                cu_seqlens = mb_data["input_ids"].offsets()
                temperature = mb_data["temperature"]
                calculate_entropy = tu.get_non_tensor_data(data=mb_data, key="calculate_entropy", default=False)

                # Remove padding: extract only valid tokens using cu_seqlens
                seq_lengths = cu_seqlens.diff()
                # total_tokens = seq_lengths.sum().item()

                # Gather valid tokens from each sample in the padded micro-batch
                valid_logits_list = []
                valid_labels_list = []
                for i, length in enumerate(seq_lengths):
                    valid_logits_list.append(pred[i, :length])
                    valid_labels_list.append(labels[i, :length])
                logits_rmpad = torch.cat(valid_logits_list, dim=0)  # [total_tokens, vocab]
                labels_rmpad = torch.cat(valid_labels_list, dim=0)  # [total_tokens]

                # Temperature scaling
                logits_rmpad = logits_rmpad / temperature

                # Compute log_probs
                inplace_backward = not calculate_entropy
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=labels_rmpad,
                    inplace_backward=inplace_backward,
                )

                # Compute entropy if needed
                entropy = None
                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

                # Re-form nested tensors
                log_probs_nested = torch.nested.nested_tensor_from_jagged(
                    log_probs.squeeze(0) if log_probs.dim() > 1 and log_probs.shape[0] == 1 else log_probs,
                    cu_seqlens,
                )

                model_output = {"log_probs": log_probs_nested}
                if entropy is not None:
                    model_output["entropy"] = entropy

                # Call verl's loss function
                if loss_function is not None:
                    loss, metrics = loss_function(
                        model_output=model_output, data=mb_data, dp_group=self.get_data_parallel_group()
                    )
                else:
                    loss = torch.tensor(0.0, device=pred.device)
                    metrics = {}

                captured_metrics.append(metrics)
                captured_model_outputs.append(model_output)

                return loss

            # Set wrapper loss_fn on the schedule
            pp_schedule._loss_fn = _pp_wrapper_loss_fn
            pp_schedule._has_backward = not forward_only

            ctx = torch.no_grad() if forward_only else nullcontext()
            with ctx:
                with self.trainer.train_context():
                    targets, losses = (global_labels, []) if pp_has_last_stage else (None, None)
                    if pp_has_first_stage:
                        pp_schedule.step(
                            global_input_ids,
                            positions=global_position_ids,
                            attention_masks=extra_kwargs["attention_masks"],
                            target=targets,
                            losses=losses,
                            return_outputs=False,
                        )
                    else:
                        pp_schedule.step(
                            attention_masks=extra_kwargs["attention_masks"],
                            target=targets,
                            losses=losses,
                            return_outputs=False,
                        )

            # Build output dict
            if pp_has_last_stage:
                # Aggregate losses and metrics from all micro-batches
                # total_loss = sum(l.detach().item() for l in losses) / n_microbatches
                output_lst = []
                for i in range(n_microbatches):
                    output_lst.append(
                        {
                            "model_output": captured_model_outputs[i] if i < len(captured_model_outputs) else {},
                            "loss": losses[i].detach().item() if i < len(losses) else 0.0,
                            "metrics": captured_metrics[i] if i < len(captured_metrics) else {},
                        }
                    )
                return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
            else:
                # Non-last PP stages: return placeholder output
                return {
                    "model_output": {},
                    "loss": [-1.0],
                    "metrics": {},
                }

    def model_forward_step(
        self,
        *,
        inputs: torch.Tensor,
        extra_inputs: dict[str, torch.Tensor] | None = None,
        extra_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the trainer model without backward.

        For PP, uses pp_schedule.step() with loss_fn=None and return_outputs=True.
        Only the last PP stage returns logits; other stages return None.
        """
        model_parts = self.module
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            pp_schedule = self.trainer.pp_schedule
            pp_has_first_stage = self.trainer.pp_has_first_stage

            # Configure for forward-only
            pp_schedule._loss_fn = None
            pp_schedule._has_backward = False

            with self.trainer.train_context():
                if pp_has_first_stage:
                    pred = pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=None,
                        losses=None,
                        return_outputs=True,
                    )
                else:
                    pred = pp_schedule.step(
                        **extra_kwargs,
                        target=None,
                        losses=None,
                        return_outputs=True,
                    )

            if pred is None:
                # Non-last PP stage: no outputs
                return None
        else:
            # Non-PP forward
            assert len(model_parts) == 1
            with self.trainer.train_context():
                with self.trainer.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)

        if isinstance(pred, DTensor):
            pred = pred.full_tensor()
        return pred

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.module for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            logger.warning(f"grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        """Advance learning rate scheduler."""
        self.lr_scheduler.step()
        lr = self.lr_scheduler.schedulers[0].get_last_lr()[0]
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Move model and/or optimizer to CPU or GPU."""
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            return

        device_name = get_device_name()
        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                for module in self.module:
                    load_fsdp_model_to_gpu(module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                for module in self.module:
                    offload_fsdp_model_to_cpu(module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        if max_ckpt_to_keep is not None:
            self.checkpointer.keep_latest_k = max_ckpt_to_keep

        self.checkpointer.save(curr_step=global_step)

        torch.distributed.barrier()
        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """Load checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        # Extract step number from path (verl uses global_step_N format)
        match = re.search(r"global_step_(\d+)", local_path)
        if match:
            step = int(match.group(1))
            self.checkpointer.load(step=step)
        else:
            # Fallback to latest
            self.checkpointer.load(step=-1)

        torch.distributed.barrier()
        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(self, **kwargs):
        for module in self.module:
            load_fsdp_model_to_gpu(module)

        # Collect state dicts from all model parts
        params = {}
        for module in self.module:
            module_params = get_model_state_dict(module)
            params.update(module_params)

        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

        # Convert TorchTitan key names to HuggingFace key names (expected by vLLM)
        sd_adapter = self.checkpointer.sd_adapter
        if sd_adapter is not None:
            params = sd_adapter.to_hf(params)

        # When weight tying is enabled, the sd_adapter skips lm_head.weight during
        # to_hf() conversion (since it's the same tensor as embed_tokens.weight in
        # the torchtitan model). But vLLM needs lm_head.weight explicitly, so we
        # add it back as a reference to embed_tokens.weight.
        if "model.embed_tokens.weight" in params and "lm_head.weight" not in params:
            params["lm_head.weight"] = params["model.embed_tokens.weight"]

        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
        per_tensor_param = (
            (
                name,
                param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                if isinstance(param, DTensor)
                else param,
            )
            for name, param in params.items()
        )
        # TODO: support Torchtitan PEFT
        return per_tensor_param, None


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: TorchTitanEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanEngine)
        super().__enter__()
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanEngine)

        # Reshard the root FSDP module
        if self.engine.engine_config.data_parallel_shard_size > 1:
            for module in self.engine.module:
                module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: TorchTitanEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanEngine)
        super().__enter__()
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["torchtitan"], device=["cuda", "npu"])
class TorchTitanEngineWithLMHead(TorchTitanEngine):
    """TorchTitan engine implementation for language models with LM head."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        output_args = {}

        if use_remove_padding:
            input_ids = input_ids.values().unsqueeze(0)
            if position_ids.dim() == 3:
                position_ids = position_ids.values().unsqueeze(1)
            else:
                position_ids = position_ids.values().unsqueeze(0)

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            attn_type = self.trainer.model_config.layer.attention.attn_backend
            attention_mask = get_attention_masks(
                input_batch=input_ids,
                positions=position_ids,
                attn_type=attn_type,
            )
        else:
            loss_mask = micro_batch["loss_mask"]
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            max_seq_len = max(input_ids.offsets().diff())

            labels = torch.roll(input_ids.values(), shifts=-1, dims=0)
            input_ids = torch.nested.to_padded_tensor(
                input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
            )

            if position_ids.dim() == 3:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                ).transpose(0, 1)
            else:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                )

            attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
            attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
            attention_mask = torch.nested.to_padded_tensor(
                attention_mask, padding=0, output_size=(batch_size, max_seq_len)
            )

        extra_inputs = {
            "positions": position_ids,
        }
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {"attention_masks": attention_mask}
        if self.parallel_dims.cp_enabled:
            input_ids, labels, extra_kwargs = prepare_context_parallel_input(
                input_ids,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.trainer.device,
                self.trainer.config.parallelism.context_parallel_load_balancer,
            )

        # TODO(jessicazhong): multimodal is not yet supported for Torchtitan engine
        extra_inputs.update(multi_modal_inputs)
        output_args["labels"] = labels
        return input_ids, extra_inputs, extra_kwargs, output_args

    def prepare_model_outputs(self, logits, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        temperature = micro_batch["temperature"]
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        labels = output_args["labels"]
        model_output = {}

        input_ids = micro_batch["input_ids"]
        cu_seqlens = input_ids.offsets()
        if use_remove_padding:
            labels = labels.squeeze(0)
            logits_rmpad = logits.squeeze(0)
            # PyTorch's autograd doesn't allow in-place modification of views when gradients need to flow back
            logits_rmpad = logits_rmpad / temperature

            inplace_backward = True
            if calculate_entropy:
                inplace_backward = False
            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=labels,
                inplace_backward=inplace_backward,
            )

            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                else:
                    entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

            log_probs = torch.nested.nested_tensor_from_jagged(log_probs.squeeze(0), cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
        else:
            logits.div_(temperature)
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            log_probs = logprobs_from_logits(logits=logits_rmpad, labels=output_args["labels"])
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            logits = self.model_forward_step(inputs=input_ids, extra_inputs=extra_inputs, extra_kwargs=extra_kwargs)

            model_output = self.prepare_model_outputs(logits=logits, output_args=output_args, micro_batch=micro_batch)

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output
