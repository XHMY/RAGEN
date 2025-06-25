"""
Parallel environment rollout using joblib.
This module provides a joblib-based parallel implementation for environment sampling.

Author: Claude Code
Date: 2025-01-06
"""
import os
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed
import threading
import queue
from contextlib import contextmanager
import torch

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from .ctx_manager import ContextManager
from .es_manager import EnvStatus
from verl import DataProto
import random
import PIL.Image


@dataclass 
class WorkerEnvConfig:
    """Configuration for a worker's environment subset"""
    worker_id: int
    env_start_idx: int
    env_end_idx: int
    env_configs: List[Dict[str, Any]]
    mode: str = "train"


class SharedLLMInferenceManager:
    """Thread-safe LLM inference manager for parallel workers"""
    
    def __init__(self, actor_wg, tokenizer):
        self.actor_wg = actor_wg
        self.tokenizer = tokenizer
        self._lock = threading.Lock()
        self._request_queue = queue.Queue()
        self._response_cache = {}
        
    def inference_batch(self, lm_inputs: DataProto) -> DataProto:
        """Thread-safe batch inference"""
        with self._lock:
            # Directly call the actor worker group for inference
            return self._generate_sequences(lm_inputs)
    
    def _generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """Internal method to generate sequences - should be thread-safe"""
        from .agent_proxy import VllmWrapperWg, ApiCallingWrapperWg
        from verl.single_controller.ray.base import RayWorkerGroup
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        
        if isinstance(self.actor_wg, RayWorkerGroup):
            padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
            padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
            lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
            lm_outputs.meta_info = lm_inputs.meta_info
            lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
        elif isinstance(self.actor_wg, (VllmWrapperWg, ApiCallingWrapperWg)):
            lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
        else:
            raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")
        
        return lm_outputs


def parallel_env_worker(worker_config: WorkerEnvConfig, 
                       sys_config: Any,
                       tokenizer: Any,
                       shared_llm_manager: SharedLLMInferenceManager,
                       dataproto_meta_info: Dict[str, Any],
                       val: bool = False) -> List[Dict[str, Any]]:
    """
    Worker function that handles a subset of environments in parallel.
    
    Args:
        worker_config: Configuration for this worker's environment subset
        sys_config: System configuration
        tokenizer: Tokenizer for the LLM
        shared_llm_manager: Shared LLM inference manager
        dataproto_meta_info: Meta info for DataProto
        val: Whether this is validation mode
        
    Returns:
        List of rollout data from assigned environments
    """
    print(f"Worker {worker_config.worker_id}: Starting with {len(worker_config.env_configs)} environments")
    
    try:
        # Initialize context manager
        print(f"Worker {worker_config.worker_id}: Initializing context manager...")
        ctx_manager = ContextManager(sys_config, tokenizer, mode=worker_config.mode)
        print(f"Worker {worker_config.worker_id}: Context manager initialized")
        
        # Create environment instances locally within this worker
        worker_envs = []
        print(f"Worker {worker_config.worker_id}: Creating {len(worker_config.env_configs)} environments...")
        
        for i, env_config in enumerate(worker_config.env_configs):
            env_id = worker_config.env_start_idx + i
            print(f"Worker {worker_config.worker_id}: Creating environment {env_id} with config: {env_config['env_class']}")
            
            env_obj = REGISTERED_ENVS[env_config['env_class']](env_config['env_config_obj'])
            
            # Handle Alfworld special case
            if env_config.get('alfworld_game_file'):
                env_obj.assign_game_file(env_config['alfworld_game_file'])
                
            entry = {
                'tag': env_config['tag'],
                'group_id': env_id // sys_config.es_manager[worker_config.mode].group_size,
                'env_id': env_id,
                'env': env_obj,
                'config': env_config['env_config_obj'],
                'status': EnvStatus(),
                'max_actions_per_traj': env_config['max_actions_per_traj']
            }
            worker_envs.append(entry)
        
        print(f"Worker {worker_config.worker_id}: All environments created successfully")
        
    except Exception as e:
        print(f"Worker {worker_config.worker_id}: Error during environment creation: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    try:
        # Initialize rollout cache
        print(f"Worker {worker_config.worker_id}: Initializing rollout cache...")
        rollout_cache = []
        for entry in worker_envs:
            cache_entry = {
                "env_id": entry['env_id'], 
                "history": [], 
                "group_id": entry['group_id'], 
                "tag": entry['tag'], 
                "penalty": 0
            }
            rollout_cache.append(cache_entry)
        
        # Reset environments
        print(f"Worker {worker_config.worker_id}: Resetting environments...")
        if worker_config.mode == "train":
            base_seed = random.randint(0, 1000000)
        else:
            base_seed = 123
            
        for i, entry in enumerate(worker_envs):
            seed = base_seed + worker_config.worker_id * 1000 + i
            print(f"Worker {worker_config.worker_id}: Resetting environment {entry['env_id']} with seed {seed}")
            entry['env'].reset(seed=seed, mode=worker_config.mode)
            entry['status'] = EnvStatus(seed=seed)
            
            # Initialize rollout cache with first observation
            cache = rollout_cache[i]
            next_state = _handle_mm_state(entry['env'].render())
            cache['history'] = _update_cache_history(
                cache['history'], 
                next_state=next_state, 
                actions_left=entry['max_actions_per_traj'], 
                num_actions_info=None
            )
        
        print(f"Worker {worker_config.worker_id}: Environment reset completed")
        
    except Exception as e:
        print(f"Worker {worker_config.worker_id}: Error during environment reset: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    # Run rollout for max_turn iterations
    max_turn = sys_config.agent_proxy.max_turn
    active_env_indices = list(range(len(worker_envs)))
    
    for turn in range(max_turn):
        if not active_env_indices:
            break
            
        # Prepare active environment outputs for this turn
        active_env_outputs = []
        for idx in active_env_indices:
            active_env_outputs.append(rollout_cache[idx])
        
        if not active_env_outputs:
            break
        
        # Get LLM inputs using context manager
        lm_inputs = ctx_manager.get_lm_inputs(active_env_outputs, prepare_for_update=False)
        lm_inputs.meta_info = dataproto_meta_info or {}
        
        # Generate sequences using shared LLM manager
        lm_outputs = shared_llm_manager.inference_batch(lm_inputs)
        
        # Get environment inputs
        env_inputs = ctx_manager.get_env_inputs(lm_outputs)
        
        # Step active environments
        new_active_indices = []
        for env_input in env_inputs:
            env_id = env_input['env_id']
            # Find the local index for this env_id
            local_idx = None
            for i, entry in enumerate(worker_envs):
                if entry['env_id'] == env_id:
                    local_idx = i
                    break
            
            if local_idx is None:
                continue
                
            entry = worker_envs[local_idx]
            cache = rollout_cache[local_idx]
            
            # Execute actions
            turn_done = _step_single_env(entry, cache, env_input, sys_config)
            
            # Keep environment active if not done
            if not turn_done and local_idx in active_env_indices:
                new_active_indices.append(local_idx)
        
        active_env_indices = new_active_indices
    
    # Finalize rollout states
    final_rollout_data = []
    for i, (entry, cache) in enumerate(zip(worker_envs, rollout_cache)):
        _finalize_rollout_metrics(entry, cache)
        final_rollout_data.append(cache)
        
        # Close environment
        entry['env'].close()
    
    return final_rollout_data


def _handle_mm_state(state: Union[str, np.ndarray, List[np.ndarray]]):
    """Handle multimodal state - copied from es_manager.py"""
    import PIL.Image
    
    if isinstance(state, str):
        return state
    elif isinstance(state, np.ndarray):
        state = [state]
    results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
    return results


def _update_cache_history(history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
    """Update cache history - copied from es_manager.py"""
    if num_actions_info is not None:
        assert len(history), "History should not be empty"
        history[-1].update(num_actions_info)
    
    entry = {}
    if isinstance(next_state, str):
        entry['state'] = next_state
    else:
        entry['state'] = "<images>" * len(next_state)
        entry['images'] = next_state
    entry['actions_left'] = actions_left
    history.append(entry)
    return history


def _step_single_env(entry: Dict, cache: Dict, env_input: Dict, sys_config: Any) -> bool:
    """Step a single environment and update its cache"""
    env = entry['env']
    status = entry['status']
    actions_left_before = entry['max_actions_per_traj'] - status.num_actions
    
    # Extract and map valid actions
    valid_actions = _extract_map_valid_actions(entry, env_input['actions'])
    
    # Execute actions
    acc_reward, turn_info, turn_done = 0, {}, False
    executed_actions = []
    
    for action in valid_actions[:actions_left_before]:
        _, reward, done, info = env.step(action)
        acc_reward += reward
        turn_info.update(info)
        executed_actions.append(action)
        if done:
            turn_done = True
            break
    
    # Apply format penalty if needed
    if len(valid_actions) != len(env_input['actions']) or not valid_actions:
        cache["penalty"] += sys_config.es_manager.format_penalty
    
    # Update status and cache
    obs = _handle_mm_state(env.render())
    status.num_actions += len(executed_actions)
    status.rewards.append(acc_reward)
    actions_left = entry['max_actions_per_traj'] - status.num_actions
    
    if turn_done:
        status.terminated = True
        status.truncated = not turn_info.get('success', False)
    
    if status.num_actions >= entry['max_actions_per_traj'] and not turn_done:
        status.truncated = True
        status.terminated = True
        turn_done = True
    
    cache['history'] = _update_cache_history(
        cache['history'], 
        next_state=obs, 
        actions_left=actions_left, 
        num_actions_info={
            'actions': executed_actions, 
            'reward': acc_reward, 
            'info': turn_info,
            'llm_response': env_input['llm_response'], 
            'llm_raw_response': env_input['llm_raw_response']
        }
    )
    
    return turn_done


def _extract_map_valid_actions(entry: Dict, actions: List[str]) -> List[str]:
    """Extract valid actions from action lookup table - copied from es_manager.py"""
    mapped_actions = []
    action_lookup = getattr(entry['config'], 'action_lookup', None)
    if action_lookup is None:
        mapped_actions = actions
    else:
        rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
        actions = [action.lower() for action in actions]
        mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
    return mapped_actions


def _finalize_rollout_metrics(entry: Dict, cache: Dict):
    """Finalize rollout metrics - adapted from es_manager.py"""
    status = entry['status']
    TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']
    
    env_metric = {
        'success': float(status.terminated and (not status.truncated)),
        'num_actions': status.num_actions,
    }
    
    custom_metric = {}
    for turn in cache['history']:
        for k, v in turn.get('info', {}).items():
            if k == 'success':
                continue
            if k not in custom_metric:
                custom_metric[k] = []
            custom_metric[k].append(float(v))
    
    for k, v in custom_metric.items():
        if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
            env_metric[k] = np.sum(v) / (len(cache['history']) - 1) if len(cache['history']) > 1 else 0
        else:
            env_metric[k] = np.sum(v)
    
    cache['history'][-1]['metrics'] = custom_metric
    env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
    cache['metrics'] = env_metric
    
    if entry['tag'] == "MetamathQA":
        cache['correct_answer'] = entry['env'].correct_answer


class ParallelRolloutManager:
    """Manager for parallel environment rollouts using joblib"""
    
    def __init__(self, config, actor_rollout_wg, tokenizer, n_jobs: int = -1):
        self.config = config
        self.tokenizer = tokenizer
        self.n_jobs = n_jobs if n_jobs != -1 else min(os.cpu_count(), 8)  # Reasonable default
        
        # Create shared LLM inference manager
        self.shared_llm_manager = SharedLLMInferenceManager(actor_rollout_wg, tokenizer)
        
    def parallel_rollout(self, dataproto: DataProto, val: bool = False) -> DataProto:
        """
        Execute parallel rollout across multiple environments using joblib.
        
        Args:
            dataproto: DataProto with meta_info for LLM generation
            val: Whether this is validation mode
            
        Returns:
            DataProto with rollout results
        """
        mode = "val" if val else "train"
        es_config = getattr(self.config.es_manager, mode)
        
        # Prepare environment configurations for workers
        worker_configs = self._prepare_worker_configs(es_config, mode)
        
        # Execute parallel rollouts
        start_time = time.time()
        
        # Ensure dataproto.meta_info is not None
        meta_info = dataproto.meta_info if dataproto.meta_info is not None else {}
        
        # Use joblib to run workers in parallel with timeout
        print(f"Starting {len(worker_configs)} workers with joblib backend='loky'...")
        
        # Try with 'loky' backend first (more stable for complex objects)
        try:
            worker_results = Parallel(n_jobs=self.n_jobs, backend='loky', timeout=300)(
                delayed(parallel_env_worker)(
                    worker_config=worker_config,
                    sys_config=self.config,
                    tokenizer=self.tokenizer,
                    shared_llm_manager=self.shared_llm_manager,
                    dataproto_meta_info=meta_info,
                    val=val
                ) for worker_config in worker_configs
            )
        except Exception as loky_error:
            print(f"Loky backend failed: {loky_error}, trying threading backend...")
            worker_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(parallel_env_worker)(
                    worker_config=worker_config,
                    sys_config=self.config,
                    tokenizer=self.tokenizer,
                    shared_llm_manager=self.shared_llm_manager,
                    dataproto_meta_info=meta_info,
                    val=val
                ) for worker_config in worker_configs
            )
        
        end_time = time.time()
        print(f"Parallel rollout completed in {end_time - start_time:.2f} seconds with {self.n_jobs} workers")
        
        # Aggregate results from all workers
        all_rollout_data = []
        for worker_result in worker_results:
            all_rollout_data.extend(worker_result)
        
        # Sort by env_id to maintain original order
        all_rollout_data.sort(key=lambda x: x['env_id'])
        
        # Convert to DataProto format using ContextManager
        ctx_manager = ContextManager(self.config, self.tokenizer, mode=mode)
        rollouts = ctx_manager.formulate_rollouts(all_rollout_data)
        
        return rollouts
    
    def _prepare_worker_configs(self, es_config, mode: str) -> List[WorkerEnvConfig]:
        """Prepare configuration for each worker"""
        # Get environment setup info
        total_envs = es_config.env_groups * es_config.group_size
        envs_per_worker = max(1, total_envs // self.n_jobs)
        
        # Prepare environment configurations (similar to es_manager._init_env_instances)
        env_configs = []
        done_groups = 0
        
        # Handle case where n_groups might be missing
        if not hasattr(es_config.env_configs, 'n_groups') or es_config.env_configs.n_groups is None:
            print(f"Warning: n_groups not specified in env_configs, using default distribution")
            n_groups = [es_config.env_groups // len(es_config.env_configs.tags)] * len(es_config.env_configs.tags)
        else:
            n_groups = es_config.env_configs.n_groups
        
        # Pre-load Alfworld if needed
        alfworld_game_files = None
        for tag, n_group in zip(es_config.env_configs.tags, n_groups):
            if tag == 'Alfworld':
                from ragen.env.alfworld_old.env import get_all_alfworld_game_files
                sample_config = REGISTERED_ENV_CONFIGS['alfworld_old']()
                alfworld_game_files = get_all_alfworld_game_files(sample_config.config_file)
                print(f"Loaded {len(alfworld_game_files)} Alfworld game files for parallel distribution")
                break
        
        for tag, n_group in zip(es_config.env_configs.tags, n_groups):
            cfg_template = self.config.custom_envs[tag]
            env_class = cfg_template.env_type
            max_actions_per_traj = cfg_template.max_actions_per_traj
            
            for env_id in range(done_groups * es_config.group_size, (done_groups + n_group) * es_config.group_size):
                if cfg_template.env_config is None:
                    env_config_obj = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config_obj = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                
                config_dict = {
                    'tag': tag,
                    'env_class': env_class,
                    'env_config_obj': env_config_obj,
                    'max_actions_per_traj': max_actions_per_traj
                }
                
                # Add Alfworld game file if applicable
                if env_class == 'alfworld_old' and alfworld_game_files:
                    game_file_index = env_id % len(alfworld_game_files)
                    config_dict['alfworld_game_file'] = alfworld_game_files[game_file_index]
                
                env_configs.append(config_dict)
            
            done_groups += n_group
        
        # Split environments across workers
        worker_configs = []
        for worker_id in range(self.n_jobs):
            start_idx = worker_id * envs_per_worker
            end_idx = min(start_idx + envs_per_worker, total_envs)
            
            if start_idx >= total_envs:
                break
                
            worker_env_configs = env_configs[start_idx:end_idx]
            
            worker_config = WorkerEnvConfig(
                worker_id=worker_id,
                env_start_idx=start_idx,
                env_end_idx=end_idx,
                env_configs=worker_env_configs,
                mode=mode
            )
            worker_configs.append(worker_config)
        
        return worker_configs