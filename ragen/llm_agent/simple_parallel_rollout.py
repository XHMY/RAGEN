"""
Simplified parallel environment rollout for debugging.
This version avoids complex object sharing and runs everything locally within workers.
"""
import os
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed
import random
import PIL.Image

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from .ctx_manager import ContextManager
from .es_manager import EnvStatus
from verl import DataProto
import threading


class SharedLLMManager:
    """Thread-safe LLM manager that can be shared across workers"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._llm = None
        self._lock = threading.Lock()
        
    def _get_llm(self):
        """Lazy initialization of LLM instance"""
        if self._llm is None:
            with self._lock:
                if self._llm is None:  # Double-check pattern
                    from .agent_proxy import VllmWrapperWg
                    self._llm = VllmWrapperWg(self.config, self.tokenizer)
        return self._llm
    
    def inference_batch(self, lm_inputs):
        """Thread-safe inference method"""
        llm = self._get_llm()
        with self._lock:
            return llm.generate_sequences(lm_inputs)


def _handle_mm_state(state: Union[str, np.ndarray, List[np.ndarray]]):
    """Handle multimodal state - copied from es_manager.py"""
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


def simple_worker_rollout(worker_id: int, 
                         env_configs: List[Dict],
                         sys_config: Any,
                         tokenizer_path: str,
                         dataproto_meta_info: Dict[str, Any],
                         val: bool = False) -> List[Dict[str, Any]]:
    """
    Simplified worker that runs actual rollout logic without shared objects.
    Each worker creates its own tokenizer and runs environments locally.
    """
    # Import required modules locally
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    mode = "val" if val else "train"
    
    # Initialize context manager
    ctx_manager = ContextManager(sys_config, tokenizer, mode=mode)
    
    # Create environment instances
    worker_envs = []
    for env_config in env_configs:
        env_id = env_config['env_id']
        env_obj = REGISTERED_ENVS[env_config['env_class']](env_config['env_config_obj'])
        
        # Handle Alfworld special case
        if env_config.get('alfworld_game_file'):
            env_obj.assign_game_file(env_config['alfworld_game_file'])
            
        entry = {
            'tag': env_config['tag'],
            'group_id': env_config['group_id'],
            'env_id': env_id,
            'env': env_obj,
            'config': env_config['env_config_obj'],
            'status': EnvStatus(),
            'max_actions_per_traj': env_config['max_actions_per_traj']
        }
        worker_envs.append(entry)
    
    # Initialize rollout cache
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
    if mode == "train":
        base_seed = random.randint(0, 1000000)
    else:
        base_seed = 123
        
    for i, entry in enumerate(worker_envs):
        seed = base_seed + worker_id * 1000 + i
        entry['env'].reset(seed=seed, mode=mode)
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
    
    # PARALLEL ROLLOUT: Use shared LLM inference manager
    max_turn = sys_config.agent_proxy.max_turn
    
    for turn in range(max_turn):
        # Get active environment outputs
        active_env_outputs = []
        active_indices = []
        
        for i, cache in enumerate(rollout_cache):
            if not worker_envs[i]['status'].terminated:
                active_env_outputs.append(cache)
                active_indices.append(i)
        
        if not active_env_outputs:
            break
        
        # Get LLM inputs using context manager
        lm_inputs = ctx_manager.get_lm_inputs(active_env_outputs, prepare_for_update=False)
        lm_inputs.meta_info = dataproto_meta_info
        
        # Use shared LLM manager for inference instead of local instance
        lm_outputs = shared_llm_manager.inference_batch(lm_inputs)
        
        # Get environment inputs from LLM outputs
        env_inputs = ctx_manager.get_env_inputs(lm_outputs)
        
        # Execute actions for each active environment
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
            
            # Execute actions using the same logic as es_manager
            env = entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions
            
            # Extract and map valid actions
            valid_actions = _extract_map_valid_actions(entry, env_input['actions'])
            
            # Execute actions
            acc_reward = 0
            turn_info = {}
            turn_done = False
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
            entry['status'].num_actions += len(executed_actions)
            entry['status'].rewards.append(acc_reward)
            actions_left = entry['max_actions_per_traj'] - entry['status'].num_actions
            
            if turn_done:
                entry['status'].terminated = True
                entry['status'].truncated = not turn_info.get('success', False)
            
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not turn_done:
                entry['status'].truncated = True
                entry['status'].terminated = True
            
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
    
    # Finalize rollout metrics for each environment (from es_manager logic)
    final_rollout_data = []
    for entry, cache in zip(worker_envs, rollout_cache):
        _finalize_rollout_metrics(entry, cache)
        final_rollout_data.append(cache)
        entry['env'].close()
    
    return final_rollout_data


def simple_worker_rollout_with_initialized_envs(worker_id: int,
                                               initialized_envs: Dict,
                                               sys_config: Any,
                                               tokenizer_path: str,
                                               shared_llm_manager: Any,
                                               dataproto_meta_info: Dict[str, Any],
                                               val: bool = False) -> List[Dict[str, Any]]:
    """
    Worker function that runs rollouts with pre-initialized environments and shared LLM.
    """
    # Extract pre-initialized components
    worker_envs = initialized_envs['worker_envs']
    rollout_cache = initialized_envs['rollout_cache']
    ctx_manager = initialized_envs['ctx_manager']
    
    # PARALLEL ROLLOUT: Use shared LLM inference manager
    max_turn = sys_config.agent_proxy.max_turn
    
    for turn in range(max_turn):
        # Get active environment outputs
        active_env_outputs = []
        active_indices = []
        
        for i, cache in enumerate(rollout_cache):
            if not worker_envs[i]['status'].terminated:
                active_env_outputs.append(cache)
                active_indices.append(i)
        
        if not active_env_outputs:
            break
        
        # Get LLM inputs using context manager
        lm_inputs = ctx_manager.get_lm_inputs(active_env_outputs, prepare_for_update=False)
        lm_inputs.meta_info = dataproto_meta_info
        
        # Use shared LLM manager for inference
        lm_outputs = shared_llm_manager.inference_batch(lm_inputs)
        
        # Get environment inputs from LLM outputs
        env_inputs = ctx_manager.get_env_inputs(lm_outputs)
        
        # Execute actions for each active environment
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
            
            # Execute actions using the same logic as es_manager
            env = entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions
            
            # Extract and map valid actions
            valid_actions = _extract_map_valid_actions(entry, env_input['actions'])
            
            # Execute actions
            acc_reward = 0
            turn_info = {}
            turn_done = False
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
            entry['status'].num_actions += len(executed_actions)
            entry['status'].rewards.append(acc_reward)
            actions_left = entry['max_actions_per_traj'] - entry['status'].num_actions
            
            if turn_done:
                entry['status'].terminated = True
                entry['status'].truncated = not turn_info.get('success', False)
            
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not turn_done:
                entry['status'].truncated = True
                entry['status'].terminated = True
            
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
    
    # Finalize rollout metrics for each environment
    final_rollout_data = []
    for entry, cache in zip(worker_envs, rollout_cache):
        _finalize_rollout_metrics(entry, cache)
        final_rollout_data.append(cache)
        entry['env'].close()
    
    return final_rollout_data


class SimpleParallelRolloutManager:
    """Simplified parallel rollout manager for debugging"""
    
    def __init__(self, config, tokenizer, n_jobs: int = 4):
        self.config = config
        self.tokenizer_path = config.actor_rollout_ref.model.path  # Store path instead of object
        self.n_jobs = n_jobs
        
    def parallel_rollout(self, dataproto: DataProto, val: bool = False) -> DataProto:
        """Execute simplified parallel rollout"""
        print(f"Starting simplified parallel rollout with {self.n_jobs} workers")
        
        mode = "val" if val else "train"
        es_config = getattr(self.config.es_manager, mode)
        
        # Prepare simplified environment configurations
        env_configs = self._prepare_simple_env_configs(es_config, mode)
        
        # Create shared LLM manager for all workers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        shared_llm_manager = SharedLLMManager(self.config, tokenizer)
        
        # Split environments across workers
        envs_per_worker = max(1, len(env_configs) // self.n_jobs)
        worker_env_configs = []
        
        for worker_id in range(self.n_jobs):
            start_idx = worker_id * envs_per_worker
            end_idx = min(start_idx + envs_per_worker, len(env_configs))
            
            if start_idx >= len(env_configs):
                break
                
            worker_configs = env_configs[start_idx:end_idx]
            worker_env_configs.append((worker_id, worker_configs))
        
        # Execute parallel rollouts
        start_time = time.time()
        meta_info = dataproto.meta_info if dataproto.meta_info is not None else {}
        
        print(f"Running {len(worker_env_configs)} workers...")
        
        # Initialize all environments sequentially first to avoid parser conflicts
        print("Initializing environments sequentially to avoid parser conflicts...")
        initialized_env_groups = []
        for worker_id, worker_configs in worker_env_configs:
            initialized_envs = self._initialize_worker_envs(worker_id, worker_configs, val)
            initialized_env_groups.append((worker_id, initialized_envs))
        
        # Execute parallel rollouts with shared LLM manager using pre-initialized environments
        worker_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(simple_worker_rollout_with_initialized_envs)(
                worker_id=worker_id,
                initialized_envs=initialized_envs,
                sys_config=self.config,
                tokenizer_path=self.tokenizer_path,
                shared_llm_manager=shared_llm_manager,
                dataproto_meta_info=meta_info,
                val=val
            ) for worker_id, initialized_envs in initialized_env_groups
        )
        
        end_time = time.time()
        print(f"Simplified parallel rollout completed in {end_time - start_time:.2f} seconds")
        
        # Aggregate results
        all_rollout_data = []
        for worker_result in worker_results:
            all_rollout_data.extend(worker_result)
        
        # Sort by env_id
        all_rollout_data.sort(key=lambda x: x['env_id'])
        
        # Convert to DataProto format
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        ctx_manager = ContextManager(self.config, tokenizer, mode=mode)
        rollouts = ctx_manager.formulate_rollouts(all_rollout_data)
        
        return rollouts
    
    def _initialize_worker_envs(self, worker_id: int, env_configs: List[Dict], val: bool = False):
        """Initialize environments for a worker sequentially to avoid parser conflicts"""
        import random
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        mode = "val" if val else "train"
        
        # Initialize context manager
        ctx_manager = ContextManager(self.config, tokenizer, mode=mode)
        
        # Create environment instances
        worker_envs = []
        for env_config in env_configs:
            env_id = env_config['env_id']
            env_obj = REGISTERED_ENVS[env_config['env_class']](env_config['env_config_obj'])
            
            # Handle Alfworld special case
            if env_config.get('alfworld_game_file'):
                env_obj.assign_game_file(env_config['alfworld_game_file'])
                
            entry = {
                'tag': env_config['tag'],
                'group_id': env_config['group_id'],
                'env_id': env_id,
                'env': env_obj,
                'config': env_config['env_config_obj'],
                'status': EnvStatus(),
                'max_actions_per_traj': env_config['max_actions_per_traj']
            }
            worker_envs.append(entry)
        
        # Initialize rollout cache
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
        if mode == "train":
            base_seed = random.randint(0, 1000000)
        else:
            base_seed = 123
            
        for i, entry in enumerate(worker_envs):
            seed = base_seed + worker_id * 1000 + i
            entry['env'].reset(seed=seed, mode=mode)
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
        
        return {
            'worker_envs': worker_envs,
            'rollout_cache': rollout_cache,
            'ctx_manager': ctx_manager
        }
    
    def _prepare_simple_env_configs(self, es_config, mode: str) -> List[Dict]:
        """Prepare simplified environment configurations"""
        env_configs = []
        
        # Handle missing n_groups
        if not hasattr(es_config.env_configs, 'n_groups') or es_config.env_configs.n_groups is None:
            n_groups = [es_config.env_groups // len(es_config.env_configs.tags)] * len(es_config.env_configs.tags)
        else:
            n_groups = es_config.env_configs.n_groups
        
        done_groups = 0
        
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
                    'env_id': env_id,
                    'tag': tag,
                    'group_id': env_id // es_config.group_size,
                    'env_class': env_class,
                    'env_config_obj': env_config_obj,
                    'max_actions_per_traj': max_actions_per_traj
                }
                
                env_configs.append(config_dict)
            
            done_groups += n_group
        
        return env_configs
    
    def _create_empty_rollout_result(self, es_config, mode: str):
        """Create empty rollout result for fallback"""
        from transformers import AutoTokenizer
        import torch
        from tensordict import TensorDict
        
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # Create minimal empty result
        batch_size = es_config.env_groups * es_config.group_size
        seq_len = 100  # Minimal sequence length
        
        rollouts = DataProto()
        rollouts.batch = TensorDict({
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
            "responses": torch.zeros(batch_size, seq_len-1, dtype=torch.long),
            "loss_mask": torch.zeros(batch_size, seq_len-1, dtype=torch.bool),
            "rm_scores": torch.zeros(batch_size, seq_len-1, dtype=torch.float),
            "original_rm_scores": torch.zeros(batch_size, seq_len-1, dtype=torch.float),
        }, batch_size=batch_size)
        
        rollouts.non_tensor_batch = {
            "env_ids": np.arange(batch_size, dtype=object),
            "group_ids": np.arange(batch_size, dtype=object),
        }
        
        rollouts.meta_info = {"metrics": {"test/success": 0.0}}
        
        return rollouts