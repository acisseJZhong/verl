TorchTitan Backend
==================

Last updated: 07/06/2026.

We support the `TorchTitan <https://github.com/pytorch/torchtitan>`_ backend by
implementing the ``TorchTitanEngine`` and ``TorchTitanEngineWithLMHead`` engine
classes. The TorchTitan backend delegates model building, parallelization
(FSDP2 / TP / PP / CP / EP), optimizer construction and sharding, LR
scheduling, gradient clipping, and checkpointing to TorchTitan's
infrastructure, while using verl's training loop, data pipeline, and loss
function.

Enable it with ``model_engine=torchtitan``. The engine is registered for the
``language_model`` model type on ``cuda`` and ``npu`` devices.

**Requirements**

- A recent TorchTitan **nightly** (the engine uses TorchTitan's ``Trainer``,
  ``ParallelismConfig.spmd_backend``, and ``activation_checkpoint`` APIs).
- A matching PyTorch **nightly** recent enough to support the ``spmd_types``
  SPMD backend (the DTensor / ``fully_shard`` fixes it depends on).
- Attention-backend-specific dependencies (see `Attention backend`_):

  - ``flex`` — no extra dependency (torch built-in FlexAttention).
  - ``flex_flash`` — ``flash-attn-4`` / CUTE kernels, Hopper or Blackwell only.
  - ``varlen`` — FA3 (``flash_attn_interface``).

**Pros**

- FSDP2, Tensor Parallelism (TP), Pipeline Parallelism (PP), Context
  Parallelism (CP), and Expert Parallelism (EP) out of the box.

- DTensor-based SPMD backends (``full_dtensor`` / ``spmd_types``) in addition
  to the legacy ``default`` sharding.

- ``torch.compile`` support, meta-device model init, and HuggingFace
  checkpoint loading (no manual conversion for TorchTitan-supported models).

- FlexAttention backends (including the FLASH kernel on Hopper/Blackwell).

**Cons**

- Requires TorchTitan and PyTorch nightlies.

- Model coverage is limited to model families registered in TorchTitan (e.g.
  ``qwen3``, ``llama3``); the flavor is derived from the HuggingFace config.

- ``sdpa`` is not a valid language-model attention backend (use ``flex``,
  ``flex_flash``, or ``varlen``).


Configuration
-------------

The TorchTitan engine is configured under ``actor_rollout_ref.<role>.torchtitan``
(mapped from ``TorchtitanEngineConfig``). Key options:

Parallelism
^^^^^^^^^^^

- ``data_parallel_shard_size`` — FSDP2 shard degree.
- ``data_parallel_replicate_size`` — HSDP replicate degree.
- ``tensor_parallel_size`` — TP degree.
- ``pipeline_parallel_size`` — PP degree.
- ``context_parallel_size`` — CP degree.
- ``expert_parallel_size`` — EP degree (MoE models).

SPMD backend
^^^^^^^^^^^^

``spmd_backend`` selects how sharding is expressed (default ``spmd_types``):

- ``default`` — legacy per-parallelism sharding (no full-DTensor mesh).
- ``full_dtensor`` — all params/buffers/inputs are DTensors on a dense
  multi-axis mesh.
- ``spmd_types`` — ``spmd_types`` typed collectives on a dense mesh.

Attention backend
^^^^^^^^^^^^^^^^^

``attn_type`` selects the attention implementation (default ``flex``):

- ``flex`` — FlexAttention; needs ``torch.compile`` to be fast (eager is slow).
- ``flex_flash`` — FlexAttention FLASH kernel; needs ``flash-attn-4`` / CUTE,
  Hopper/Blackwell only.
- ``varlen`` — fast eager, flash-style; needs FA3 (``flash_attn_interface``).

Activation checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^

``activation_checkpoint`` selects the AC mode (default ``selective``):

- ``selective`` — TorchTitan's selective (per-op) activation checkpointing.
- ``full`` — full activation checkpointing.
- ``none`` — disabled.

.. note::

   Under ``spmd_backend=spmd_types``, TorchTitan's typed collectives require the
   thread-local SPMD mesh to stay active across ``loss.backward()`` (activation
   checkpointing re-runs the module forward during backward). The engine mirrors
   TorchTitan's ``init_distributed`` by disabling autograd multithreading at
   device-mesh init so the recompute runs on the calling thread and can access
   the mesh. Without this (e.g. when calling backward from a non-main thread as
   verl does under Ray), ``spmd.assert_type`` would raise
   ``SpmdTypeError: ... no current mesh is set``.


PPO Example
-----------

An end-to-end GRPO example on GSM8K with the TorchTitan engine is provided at
`tests/special_e2e/run_ppo_trainer_torchtitan.sh <https://github.com/verl-project/verl/blob/main/tests/special_e2e/run_ppo_trainer_torchtitan.sh>`_.

Basic: Qwen3-0.6B with FSDP2 + spmd_types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qwen3-0.6B, pure FSDP across 4 GPUs, ``flex`` attention, ``spmd_types`` backend,
selective activation checkpointing:

.. code:: shell

   NUM_GPUS=4 FSDP_SIZE=4 ATTN_TYPE=flex SPMD_BACKEND=spmd_types AC_MODE=selective \
       bash tests/special_e2e/run_ppo_trainer_torchtitan.sh

The script exposes ``NUM_GPUS``, ``FSDP_SIZE``, ``TP_SIZE``, ``EP_SIZE``,
``ATTN_TYPE``, ``SPMD_BACKEND``, and ``AC_MODE`` as environment variables.

Adding tensor parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^

To mirror ``FSDP_SIZE=2 TP_SIZE=2`` on 4 GPUs:

.. code:: shell

   NUM_GPUS=4 FSDP_SIZE=2 TP_SIZE=2 ATTN_TYPE=flex SPMD_BACKEND=spmd_types \
       bash tests/special_e2e/run_ppo_trainer_torchtitan.sh
