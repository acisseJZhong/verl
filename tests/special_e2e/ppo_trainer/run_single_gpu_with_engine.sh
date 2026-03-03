set -xeuo pipefail

# Test 2: Multi-GPU FSDP2 with Qwen3-30B-A3B (MoE), FSDP8
QWEN30B_MODEL_ID=Qwen/Qwen3-30B-A3B
QWEN30B_MODEL_PATH=${HOME}/models/${QWEN30B_MODEL_ID}

GRPC_ENABLE_FORK_SUPPORT=0 NCCL_NVLS_ENABLE=0 \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=16  \
  data.max_prompt_length=512 \
  data.max_response_length=2048  \
  data.seed=42 \
  actor_rollout_ref.model.path="${QWEN30B_MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.zero_indexed_step=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1  \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=2 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.val_before_train=False \
  trainer.use_legacy_worker_impl=disable \
  trainer.logger=['console','file','wandb'] \
  trainer.project_name='verl_grpo_example_gsm8k_0302' \
  trainer.experiment_name="qwen3-30b-a3b-fsdp2-fsdp8" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps=100
