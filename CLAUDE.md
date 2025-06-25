# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Training
```bash
# Basic training with default config
python train.py --config-name base

# Training with LoRA
python train.py --config-name base-lora

### Evaluation
```bash
# Evaluate a model
python -m ragen.llm_agent.agent_proxy --config-name <eval_config>

# Direct evaluation scripts
python ragen/eval.py
python ragen/eval_api.py
```

## Architecture Overview

RAGEN is a reinforcement learning framework for training LLM reasoning agents in interactive environments. The system implements the StarPO (State-Thinking-Action-Reward Policy Optimization) algorithm.

### Core Components

**Three-Module Design:**
- **Environment State Manager** (`ragen/llm_agent/es_manager.py`): Manages multiple environments, records states during rollout, processes actions in batch-wise manner
- **Context Manager** (`ragen/llm_agent/ctx_manager.py`): Parses agent tokens into structured actions, formats observations, compiles trajectories for LLM updating
- **Agent Proxy** (`ragen/llm_agent/agent_proxy.py`): Interface for executing single/multi-round rollouts

**Training Infrastructure:**
- **Agent Trainer** (`ragen/trainer/agent_trainer.py`): Ray-based distributed training with FSDP support
- **Core Algorithms** (`ragen/trainer/core_algos.py`): PPO/GRPO implementations, advantage computation (GAE, bi-level GAE)
- **Workers** (`ragen/workers/`): Distributed actor/critic workers with FSDP sharding

### Configuration System

Hydra-based configuration with inheritance:
- `config/base.yaml`: Main config inheriting from `ppo_trainer.yaml` and `envs.yaml`
- `config/base-lora.yaml`: LoRA-enabled training config
- Environment configs in `config/envs.yaml`
- Individual environment configs: `config/_1_bandit.yaml`, `config/_2_sokoban.yaml`, etc.

### Key Features

**Multi-turn RL Training:** Agents perform sequential decision-making across multiple interaction turns
**Stochastic Environment Support:** Handles uncertainty where identical actions can lead to different outcomes
**Reward Normalization:** Multiple strategies (state/batch/inductive grouping, identity/mean_std methods)
**Distributed Training:** Ray-based with FSDP for memory efficiency
**Modular Environment System:** Easy addition of new environments via OpenAI Gym interface

## Adding New Environments

1. Implement environment in `ragen/env/new_env/env.py` with required methods: `step()`, `reset()`, `render()`, `close()`
2. Define config in `ragen/env/new_env/config.py`
3. Register in `config/envs.yaml` under `custom_envs`
4. Add environment tag to `es_manager` section in `config/base.yaml`

## Dependencies

Use `conda activate verl` to activate the environment.