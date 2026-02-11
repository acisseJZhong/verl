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
import logging
import os
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.pipelining.schedules import get_schedule_class
from torchtitan.config.job_config import Checkpoint, Compile, JobConfig, LRScheduler, Model, Optimizer, Parallelism
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
from verl.workers.config import TorchtitanEngineConfig, TorchtitanModelConfig, TorchtitanOptimizerConfig
from verl.workers.engine.torchtitan.utils import get_attention_masks

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches

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
        model_config: TorchtitanModelConfig,
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

        # Disable torchtitan's dataloader since verl has its own data loading
        # Ideally torchtitan trainer init should not initialize dataloader
        import torchtitan.protocols.train_spec as train_spec_module

        original_get_train_spec = train_spec_module.get_train_spec

        def _get_train_spec_without_dataloader(model_name):
            train_spec = original_get_train_spec(model_name)
            train_spec.build_dataloader_fn = None
            return train_spec

        train_spec_module.get_train_spec = _get_train_spec_without_dataloader

        # Get train_spec and directly override model_args before Trainer init
        train_spec = train_spec_module.get_train_spec(self.model_config.name)
        model_args = train_spec.model_args.get(self.model_config.flavor)
        if model_args is not None:
            if hasattr(model_args, "attn_type"):
                model_args.attn_type = self.model_config.attn_type
            if hasattr(model_args, "attn_mask_type"):
                model_args.attn_mask_type = self.model_config.attn_mask_type

        model = Model(
            name=self.model_config.name,
            flavor=self.model_config.flavor,
            hf_assets_path=self.model_config.hf_assets_path,
        )
        optimizer = Optimizer(
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

        lr_scheduler = LRScheduler(
            warmup_steps=lr_warmup_steps,
            decay_type=self.optimizer_config.decay_type,
            min_lr_factor=self.optimizer_config.min_lr_factor,
        )
        parallelism = Parallelism(
            data_parallel_replicate_degree=self.engine_config.data_parallel_replicate_size,
            data_parallel_shard_degree=self.engine_config.data_parallel_shard_size,
            fsdp_reshard_after_forward=self.engine_config.reshard_after_forward,
            tensor_parallel_degree=self.engine_config.tensor_parallel_size,
            pipeline_parallel_degree=self.engine_config.pipeline_parallel_size,
            pipeline_parallel_schedule=self.engine_config.pipeline_parallel_schedule,
            pipeline_parallel_layers_per_stage=self.engine_config.pipeline_parallel_layers_per_stage,
            pipeline_parallel_first_stage_less_layers=self.engine_config.pipeline_parallel_first_stage_less_layers,
            pipeline_parallel_last_stage_less_layers=self.engine_config.pipeline_parallel_last_stage_less_layers,
            context_parallel_degree=self.engine_config.context_parallel_size,
            expert_parallel_degree=self.engine_config.expert_parallel_size,
            expert_tensor_parallel_degree=self.engine_config.expert_tensor_parallel_size,
        )
        checkpoint = Checkpoint(
            enable=True,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            initial_load_path=model_config.hf_assets_path,
        )
        compile = Compile(enable=self.engine_config.use_torch_compile)

        # Construct Torchtitan's JobConfig
        self.config = JobConfig(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallelism=parallelism,
            checkpoint=checkpoint,
            compile=compile,
        )
        self.trainer = Trainer(self.config)

        self._init_device_mesh()

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
        mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        return 0 if mesh is None else mesh.get_local_rank()

    def get_data_parallel_size(self):
        mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        return 1 if mesh is None else mesh.size()

    def get_data_parallel_group(self):
        mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        if mesh is None:
            return None
        return mesh.get_group()

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """Perform forward and optionally backward pass on a batch."""
        tu.assign_non_tensor(data, sp_size=self.engine_config.tensor_parallel_size)

        # Compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())
        ctx = torch.no_grad() if forward_only else nullcontext()
        output_lst = []

        if self.parallel_dims.pp_enabled:
            num_batches_divided_by = None
            if self.engine_config.pipeline_parallel_layers_per_stage is not None:
                num_stages = self.trainer.model_args.n_layers // self.engine_config.pipeline_parallel_layers_per_stage
                vpp_size = num_stages // self.parallel_dims.pp
                if vpp_size > 1:
                    # todo(jessicazhong): double check if this correctly maps with megatron PP
                    num_batches_divided_by = num_stages

            micro_batches, indices = prepare_micro_batches(
                data=data,
                dp_group=self.get_data_parallel_group(),
                num_batches_divided_by=num_batches_divided_by,
                same_micro_num_in_dp=True,
                min_num_micro_batch=None,
            )

            if num_batches_divided_by is not None:
                assert len(micro_batches) % num_batches_divided_by == 0, (
                    f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                    f"{num_batches_divided_by} for for interleaved PP schedule."
                )

            # compute input shapes for pp stages
            n_micro_batch = len(micro_batches)
            for micro_batch in micro_batches:
                tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)

            loss, output = self.forward_step(micro_batches, loss_function=loss_function, forward_only=forward_only)
            output_lst.append(output)
        else:
            micro_batches, indices = prepare_micro_batches(
                data=data,
                dp_group=self.get_data_parallel_group(),
                same_micro_num_in_dp=True,
            )
            for micro_batch in micro_batches:
                with ctx:
                    loss, output = self.forward_step(
                        micro_batch, loss_function=loss_function, forward_only=forward_only
                    )
                    if not forward_only:
                        loss.backward()
                output_lst.append(output)

        # Only the rank with outputs (last PP stage, first TP/CP rank) processes results
        if self.is_mp_src_rank_with_outputs():
            return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
        else:
            return {}

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
            print(f"WARN: grad_norm is not finite: {grad_norm}")
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
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
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
            attn_type = self.trainer.model_args.attn_type
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
                self.trainer.job_config.parallelism.context_parallel_load_balancer,
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
            logits_rmpad.div_(temperature)

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

            # TODO(jessicazhong): how to handle this with TorchTitan SP
            # if self.use_ulysses_sp:
            #     pad_size = output_args["pad_size"]

            #     log_probs = gather_outputs_and_unpad(
            #         log_probs,
            #         gather_dim=0,
            #         unpad_dim=0,
            #         padding_size=pad_size,
            #     )
            #     if calculate_entropy:
            #         entropy_rmpad = gather_outputs_and_unpad(
            #             entropy_rmpad,
            #             gather_dim=0,
            #             unpad_dim=0,
            #             padding_size=pad_size,
            #         )
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

    def forward_step(self, micro_batch: TensorDict | list[TensorDict], loss_function, forward_only):
        """Forward step that handles both single TensorDict and list of TensorDicts.

        For PP, a list of microbatches is passed and processed together by the PP schedule.
        For non-PP, a single microbatch is processed.
        """
        # Handle list of microbatches (PP case)
        if isinstance(micro_batch, list):
            return self._forward_step_pp(micro_batch, loss_function, forward_only)
        else:
            # Single microbatch case (non-PP)
            return self._forward_step(micro_batch, loss_function, forward_only)

    def _forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        """Forward step for a single microbatch (non-PP case)."""
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            assert len(self.module) == 1
            with self.trainer.train_context():
                with self.trainer.maybe_enable_amp:
                    logits = self.module[0](input_ids, **extra_inputs, **extra_kwargs)

            if self.is_mp_src_rank_with_outputs():
                model_output = self.prepare_model_outputs(
                    logits=logits, output_args=output_args, micro_batch=micro_batch
                )

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
            else:
                return None, {}

    def _forward_step_pp(self, micro_batches: list[TensorDict], loss_function, forward_only):
        """Forward step for PP case with list of microbatches.

        Uses the private _step_microbatches API to pass pre-split microbatches directly
        to the PP schedule, avoiding the split-then-merge overhead.

        Key adaptations for verl:
        1. Creates a PP-compatible loss function that computes log_probs from logits
           and uses verl's loss function signature
        2. Stores microbatch data for use in the loss function wrapper
        3. Rebuilds the PP schedule when n_microbatches changes to recompute pipeline_order
        """
        device_name = get_device_name()
        n_microbatches = len(micro_batches)

        # Rebuild PP schedule if n_microbatches changed
        # The schedule's pipeline_order is computed at init time based on n_microbatches
        # and cannot be simply overridden. We need to create a new schedule instance.
        if self.trainer.pp_schedule._n_microbatches != n_microbatches:
            schedule_class = get_schedule_class(self.config.parallelism.pipeline_parallel_schedule)
            stages = self.trainer.pp_schedule._stages
            # Create new schedule with correct n_microbatches
            # Note: loss_fn will be overridden below, so pass None here
            self.trainer.pp_schedule = schedule_class(
                stages=stages,
                n_microbatches=n_microbatches,
                loss_fn=None,
                scale_grads=False,
            )

        pp_schedule = self.trainer.pp_schedule

        # Prepare inputs for all microbatches as separate arg/kwarg tuples
        arg_mbs = []  # List of tuples: [(input_ids,), (input_ids,), ...]
        kwarg_mbs = []  # List of dicts: [{positions: ..., attention_masks: ...}, ...]
        target_mbs = []  # List of labels tensors
        microbatch_data = []  # Store microbatch TensorDicts for loss computation

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

            # Positional args as tuple
            arg_mbs.append((input_ids,))

            # Merge extra_inputs and extra_kwargs for kwargs
            mb_kwargs = {}
            mb_kwargs.update(extra_inputs)
            mb_kwargs.update(extra_kwargs)
            kwarg_mbs.append(mb_kwargs)

            # Target labels
            target_mbs.append(output_args["labels"])

            # Store microbatch data for loss computation
            microbatch_data.append(micro_batch)

        losses = [] if self.trainer.pp_has_last_stage else None

        # Pad inputs to max_seq_len across all microbatches for consistent shapes in PP
        # This is critical for PP because shape inference happens once and all microbatches
        # must have consistent shapes for send/receive buffers
        max_seq_len = max(arg_mbs[i][0].size(1) for i in range(len(arg_mbs)))
        original_seq_lens = [arg_mbs[i][0].size(1) for i in range(len(arg_mbs))]
        for i in range(len(arg_mbs)):
            input_ids = arg_mbs[i][0]
            orig_seq_len = input_ids.size(1)
            pad_size = max_seq_len - orig_seq_len
            if pad_size > 0:
                # Pad input_ids
                input_ids_padded = torch.nn.functional.pad(input_ids, (0, pad_size), value=0)
                arg_mbs[i] = (input_ids_padded,)

                # Pad positions in kwarg_mbs
                positions = kwarg_mbs[i]["positions"]
                if positions.dim() == 2:
                    # Shape: [1, seq_len]
                    positions_padded = torch.nn.functional.pad(positions, (0, pad_size), value=0)
                else:
                    # Shape: [1, num_heads, seq_len] or similar - pad last dim
                    positions_padded = torch.nn.functional.pad(positions, (0, pad_size), value=0)
                kwarg_mbs[i]["positions"] = positions_padded

                # Update attention masks for padded sequence length
                # For varlen attention, we need to update VarlenMetadata
                attention_masks = kwarg_mbs[i].get("attention_masks")
                if attention_masks is not None:
                    from torchtitan.models.attention import VarlenMetadata

                    if isinstance(attention_masks, VarlenMetadata):
                        # For VarlenMetadata, cu_seqlens defines document boundaries
                        # To pad to max_seq_len, we add the padding tokens as a new "document"
                        # This ensures the model processes max_seq_len tokens total
                        # The padded tokens form their own document and won't attend to real tokens
                        cu_seq_q = attention_masks.cu_seq_q.clone()
                        cu_seq_k = attention_masks.cu_seq_k.clone()
                        # Append max_seq_len as the new final boundary (padding document)
                        device = cu_seq_q.device
                        cu_seq_q = torch.cat(
                            [cu_seq_q, torch.tensor([max_seq_len], dtype=cu_seq_q.dtype, device=device)]
                        )
                        cu_seq_k = torch.cat(
                            [cu_seq_k, torch.tensor([max_seq_len], dtype=cu_seq_k.dtype, device=device)]
                        )
                        kwarg_mbs[i]["attention_masks"] = VarlenMetadata(
                            cu_seq_q=cu_seq_q,
                            cu_seq_k=cu_seq_k,
                            max_q=max_seq_len,  # max could be the padding span now
                            max_k=max_seq_len,
                        )

                # Pad target labels as well to maintain shape consistency
                target = target_mbs[i]
                if target.size(-1) != max_seq_len:
                    target_padded = torch.nn.functional.pad(target, (0, pad_size), value=0)
                    target_mbs[i] = target_padded

        # Create PP-compatible loss function wrapper
        # PP schedule calls loss_fn(output, target) where:
        # - output: model logits tensor
        # - target: labels tensor from target_mbs
        # We need to convert this to verl's loss function signature
        current_mb_idx = [0]  # Use list for closure mutability

        def pp_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """PP-compatible loss function that wraps verl's loss function.

            Args:
                logits: Model output logits [batch, seq_len, vocab_size] - may be padded
                labels: Target labels tensor - may be padded

            Returns:
                loss: Scalar loss tensor
            """
            mb_idx = current_mb_idx[0]
            micro_batch = microbatch_data[mb_idx]
            orig_seq_len = original_seq_lens[mb_idx]
            current_mb_idx[0] += 1

            # Unpad logits and labels to original sequence length
            logits = logits[:, :orig_seq_len, :]
            labels = labels[:, :orig_seq_len]

            # Compute log_probs from logits (similar to prepare_model_outputs)
            temperature = micro_batch["temperature"]
            input_ids = micro_batch["input_ids"]
            cu_seqlens = input_ids.offsets()

            # Process logits similar to prepare_model_outputs
            labels_squeezed = labels.squeeze(0) if labels.dim() > 1 else labels
            logits_rmpad = logits.squeeze(0) if logits.dim() > 2 else logits
            logits_rmpad = logits_rmpad / temperature

            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=labels_squeezed,
                inplace_backward=True,
            )
            log_probs_nested = torch.nested.nested_tensor_from_jagged(
                log_probs.squeeze(0) if log_probs.dim() > 1 else log_probs, cu_seqlens
            )

            model_output = {"log_probs": log_probs_nested}

            # Call verl's loss function
            if loss_function is not None:
                loss, _ = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                loss = torch.tensor(0.0, device=logits.device)

            return loss

        # Set has_backward based on forward_only flag
        pp_schedule._has_backward = not forward_only
        # Override loss_fn to use our PP-compatible wrapper
        pp_schedule._loss_fn = pp_loss_fn

        # Set the same has_backward flag for stage objects
        # Also reset metadata so shape inference runs again for each batch
        # This is needed because verl uses dynamic sequence lengths per batch
        if hasattr(pp_schedule, "_stages"):
            for stage in pp_schedule._stages:
                stage.has_backward = pp_schedule._has_backward
            # Clear runtime states and reset metadata for dynamic shapes
            for stage in pp_schedule._stages:
                stage.clear_runtime_states()
                # Reset output metadata to allow re-inference for different sequence lengths
                stage._outputs_meta = None
                # Reset input metadata for dynamic shapes
                if hasattr(stage, "inputs_meta"):
                    stage.inputs_meta = None
            # Reset initialization flags so _prepare_forward_infra runs again
            # This is needed to trigger shape inference with new batch shapes
            pp_schedule._stages_forward_initialized = False
            pp_schedule._stages_backward_initialized = False
        elif hasattr(pp_schedule, "_stage"):
            pp_schedule._stage.has_backward = pp_schedule._has_backward
            pp_schedule._stage.clear_runtime_states()
            pp_schedule._stage._outputs_meta = None
            if hasattr(pp_schedule._stage, "inputs_meta"):
                pp_schedule._stage.inputs_meta = None
            # Reset initialization flags for single-stage schedule
            pp_schedule._stage_forward_initialized = False
            pp_schedule._stage_backward_initialized = False

        with self.trainer.train_context():
            # Wrap with autocast to ensure bf16 dtype for flash attention compatibility
            with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                # Reset microbatch index for loss function
                current_mb_idx[0] = 0

                # Use private _step_microbatches API with pre-split microbatches
                # For non-first stages, arg_mbs should be None (they receive activations from prev stage)
                if self.trainer.pp_has_first_stage:
                    pp_schedule._step_microbatches(
                        arg_mbs=arg_mbs,
                        kwarg_mbs=kwarg_mbs,
                        target_mbs=target_mbs if self.trainer.pp_has_last_stage else None,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    pp_schedule._step_microbatches(
                        arg_mbs=None,
                        kwarg_mbs=kwarg_mbs,
                        target_mbs=target_mbs if self.trainer.pp_has_last_stage else None,
                        losses=losses,
                        return_outputs=False,
                    )

        if self.is_mp_src_rank_with_outputs():
            # Aggregate loss from PP schedule
            total_loss = torch.sum(torch.stack(losses)) if losses else torch.tensor(0.0, device=device_name)

            # For PP, we return aggregated metrics
            output = {
                "model_output": {},  # PP doesn't return per-microbatch outputs easily
                "loss": total_loss.detach().item(),
                "metrics": {},
            }

            return total_loss, output
        else:
            return None, {}
