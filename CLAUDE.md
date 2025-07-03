# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Use `conda activate verl` to activate the environment.

## Core Architecture

RAGEN is a reinforcement learning framework for training LLM reasoning agents using the StarPO (State-Thinking-Actions-Reward Policy Optimization) algorithm. The system implements a three-module architecture:

### Main Components

1. **Environment State Manager** (`ragen/llm_agent/es_manager.py`): Manages multiple environments simultaneously, processes actions, and returns observations
2. **Context Manager** (`ragen/llm_agent/ctx_manager.py`): Parses agent tokens into structured actions and compiles rollout trajectories for training
3. **Agent Proxy** (`ragen/llm_agent/agent_proxy.py`): Interface for executing single/multi-round rollouts and coordinating between managers

### Key Directories

- `ragen/env/`: Environment implementations (Sokoban, FrozenLake, Bandit, Countdown, MetaMathQA, WebShop, AlfWorld)
- `ragen/trainer/`: Training algorithms (PPO, GRPO, StarPO, BiLevel GAE)
- `ragen/workers/`: Distributed training workers using Ray
- `config/`: Hierarchical YAML configuration files with Hydra integration
- `verl/`: Submodule providing core RL training infrastructure

## Training Commands

### Basic Training
```bash
# Train with base configuration
python train.py --config-name base

# Train on specific environments
python train.py --config-name _2_sokoban
python train.py --config-name _7_alfworld
```

### Memory-Constrained Training
For machines with lower memory (e.g., RTX 4090):
```bash
python train.py \
  micro_batch_size_per_gpu=1 \
  ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.response_length=128
```

### Algorithm Selection
```bash
# Use GRPO (Group Relative Policy Optimization)
python train.py --config-name base algorithm.adv_estimator=grpo \
  agent_proxy.reward_normalization.method=mean_std \
  actor_rollout_ref.actor.use_kl_loss=True

# Use PPO (default)
python train.py --config-name base algorithm.adv_estimator=gae
```

## Evaluation

```bash
python -m ragen.llm_agent.agent_proxy --config-name <eval_config>
```

## Configuration System

The configuration uses Hydra with inheritance:
- `config/base.yaml`: Main configuration inheriting from `ppo_trainer.yaml` and `envs.yaml`
- Environment-specific configs: `_1_bandit.yaml`, `_2_sokoban.yaml`, etc.
- LoRA support available in `config/base-lora.yaml`

Key configuration parameters:
- `model_path`: HuggingFace model path (default: Qwen/Qwen2.5-7B-Instruct)
- `micro_batch_size_per_gpu`: Batch size per GPU
- `actor_rollout_ref.rollout.response_length`: Response length for generation
- `algorithm.adv_estimator`: Choose between 'gae' (PPO) or 'grpo'

## Development Patterns

### Agent-Environment Interaction Format
Agents generate responses in the format: `<think>...</think><ans>action</ans>` for explicit reasoning before actions.

### Multi-Turn Reasoning
The system supports sequential decision-making with environment feedback across multiple turns, optimizing entire trajectories rather than individual steps.

### Adding Custom Environments
1. Implement OpenAI Gym-compatible environment in `ragen/env/new_env/env.py`
2. Define configuration in `ragen/env/new_env/config.py`
3. Register in `config/envs.yaml`
4. Add environment tag to `config/base.yaml`

## Distributed Training

Uses Ray for distributed training with FSDP (Fully Sharded Data Parallel) support. The system supports both local VLLM and API-based LLM calls (OpenAI, Anthropic, DeepSeek).

## Data and Logging

- Training data stored in Parquet format in `data/` directory
- WandB integration for experiment tracking
- Check `val/generations` metric in WandB to see generated trajectories

## Development Branch

Current active development on `ray_env_workers` branch includes:
- Enhanced Ray-based environment workers
- Independent rollout coordination
- VLLM API server integration
- Parallel agent proxy implementation