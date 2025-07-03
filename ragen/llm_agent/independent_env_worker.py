"""
Independent Environment Worker for self-contained rollouts.
This module provides a Ray actor that manages environments and makes
direct API calls to the VLLM server, operating completely independently.

Author: Generated for RAGEN independent rollout architecture
Date: 2025-06-25
"""

import ray
import time
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .ctx_manager import ContextManager
from .base_llm import ConcurrentLLM
from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from verl import DataProto


@dataclass
class WorkerEnvironment:
    """Environment instance managed by a worker."""
    env_id: int
    tag: str
    env: Any
    config: Any
    max_actions_per_traj: int
    group_id: int = 0
    status: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    penalty: float = 0.0


@ray.remote(num_cpus=2, num_gpus=0)  # CPU-only worker
class IndependentEnvironmentWorker:
    """
    Independent Ray environment worker that handles complete rollouts.
    
    This worker:
    - Manages multiple environments
    - Makes direct API calls to VLLM server
    - Performs complete rollouts independently
    - Returns final results without coordination
    """
    
    def __init__(self, 
                 worker_id: int,
                 environment_specs: List[Dict[str, Any]],
                 llm_config: Dict[str, Any],
                 system_config: Dict[str, Any]):
        """
        Initialize independent environment worker.
        
        Args:
            worker_id: Unique identifier for this worker
            environment_specs: List of environment configurations
            llm_config: LLM API configuration (base_url, model_name, etc.)
            system_config: System configuration for context manager
        """
        self.worker_id = worker_id
        self.system_config = system_config
        self.llm_config = llm_config
        
        # Initialize environments
        self.environments = {}
        self._initialize_environments(environment_specs)
        
        # Initialize LLM client
        self.llm_client = ConcurrentLLM(
            provider="vllm_local",
            base_url=llm_config["base_url"],
            model_name=llm_config["model_name"],
            api_key=llm_config.get("api_key", "dummy_key"),
            max_concurrency=llm_config.get("max_concurrency", 4)
        )
        
        # Initialize context managers
        self.train_ctx_manager = ContextManager(system_config, None, mode="train")  # No tokenizer needed for API
        self.val_ctx_manager = ContextManager(system_config, None, mode="val")
        
        print(f"Worker {worker_id} initialized with {len(self.environments)} environments")
        
    def _initialize_environments(self, environment_specs: List[Dict[str, Any]]):
        """Initialize environments from specifications."""
        for spec in environment_specs:
            env_id = spec['env_id']
            tag = spec['tag']
            env_config_dict = spec['env_config']
            max_actions = spec['max_actions_per_traj']
            group_id = spec.get('group_id', 0)  # Extract group_id from spec
            
            # Create environment
            env_class = env_config_dict['env_type']
            if env_config_dict.get('env_config') is None:
                env_config = REGISTERED_ENV_CONFIGS[env_class]()
            else:
                env_config = REGISTERED_ENV_CONFIGS[env_class](**env_config_dict['env_config'])
                
            env_obj = REGISTERED_ENVS[env_class](env_config)
            
            # Handle Alfworld game file assignment
            if 'alfworld_game_file' in spec:
                if hasattr(env_obj, 'assign_game_file'):
                    env_obj.assign_game_file(spec['alfworld_game_file'])
                    
            # Store environment
            self.environments[env_id] = WorkerEnvironment(
                env_id=env_id,
                tag=tag,
                env=env_obj,
                config=env_config,
                max_actions_per_traj=max_actions,
                group_id=group_id  # Store group_id
            )
            
    async def run_independent_rollout(self, 
                                    seed: Optional[int] = None, 
                                    mode: str = "train",
                                    max_turns: int = 10) -> List[Dict[str, Any]]:
        """
        Run complete independent rollout for all environments.
        
        Args:
            seed: Random seed for environment reset
            mode: Mode for rollout ("train" or "val")
            max_turns: Maximum number of turns per rollout
            
        Returns:
            List of final rollout states
        """
        ctx_manager = self.val_ctx_manager if mode == "val" else self.train_ctx_manager
        
        print(f"Worker {self.worker_id} starting independent rollout (mode={mode}, max_turns={max_turns})")
        
        # Reset all environments
        self._reset_environments(seed, mode)
        
        # Get initial observations
        active_envs = list(self.environments.values())
        
        # Main rollout loop
        for turn in range(max_turns):
            if not active_envs:
                break
                
            print(f"Worker {self.worker_id}: Turn {turn + 1}, {len(active_envs)} active environments")
            
            # Prepare observations for LLM
            env_outputs = []
            for env_wrapper in active_envs:
                env_outputs.append({
                    "env_id": env_wrapper.env_id,
                    "history": env_wrapper.history,
                    "tag": env_wrapper.tag,
                    "penalty": env_wrapper.penalty
                })
            
            # Convert to messages for API
            messages_list = self._prepare_llm_inputs(env_outputs, ctx_manager)
            
            # Make API calls to LLM
            llm_responses = await self._call_llm_api(messages_list)
            
            # Process LLM responses and step environments
            active_envs = self._step_environments(active_envs, llm_responses, ctx_manager)
            
        # Collect final results
        final_results = []
        for env_wrapper in self.environments.values():
            final_state = self._get_final_environment_state(env_wrapper)
            final_results.append(final_state)
            
        print(f"Worker {self.worker_id} completed rollout with {len(final_results)} environments")
        return final_results
        
    def _reset_environments(self, seed: Optional[int], mode: str):
        """Reset all environments."""
        base_seed = seed if seed is not None else random.randint(0, 1000000)
        
        for i, env_wrapper in enumerate(self.environments.values()):
            env_seed = base_seed + i
            env_wrapper.env.reset(seed=env_seed, mode=mode)
            
            # Reset status and history
            env_wrapper.status = {
                'truncated': False,
                'terminated': False,
                'num_actions': 0,
                'rewards': [],
                'seed': env_seed
            }
            env_wrapper.penalty = 0.0
            
            # Get initial observation
            initial_obs = self._handle_multimodal_state(env_wrapper.env.render())
            env_wrapper.history = [{
                'state': initial_obs if isinstance(initial_obs, str) else "<images>" * len(initial_obs),
                'actions_left': env_wrapper.max_actions_per_traj
            }]
            
            if not isinstance(initial_obs, str):
                env_wrapper.history[0]['images'] = initial_obs
                
    def _prepare_llm_inputs(self, env_outputs: List[Dict], ctx_manager: ContextManager) -> List[List[Dict[str, str]]]:
        """Convert environment outputs to LLM API message format."""
        messages_list = []
        
        for env_output in env_outputs:
            # Use context manager to format the prompt
            # This is a simplified version - you may need to adapt based on your context manager
            history = env_output['history']
            tag = env_output['tag']
            
            # Build conversation messages
            messages = []
            
            # Add system message if needed
            system_prompt = self._get_system_prompt(tag)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            for i, turn in enumerate(history):
                if 'state' in turn:
                    # Environment observation
                    state_content = turn['state']
                    if 'actions_left' in turn:
                        state_content += f"\nActions remaining: {turn['actions_left']}"
                    messages.append({"role": "user", "content": state_content})
                    
                if 'llm_response' in turn and turn['llm_response']:
                    # Previous LLM response
                    messages.append({"role": "assistant", "content": turn['llm_response']})
                    
            messages_list.append(messages)
            
        return messages_list
        
    async def _call_llm_api(self, messages_list: List[List[Dict[str, str]]]) -> List[str]:
        """Make API calls to LLM server."""
        try:
            generation_kwargs = self.llm_config.get('generation_kwargs', {})
            
            # Use the async concurrent LLM client to avoid event loop conflicts
            results, failed_messages = await self.llm_client.run_batch_async(
                messages_list=messages_list,
                **generation_kwargs
            )
            
            if failed_messages:
                print(f"Worker {self.worker_id}: {len(failed_messages)} API calls failed")
                
            # Extract response texts
            response_texts = []
            for result in results:
                if result is not None:
                    response_texts.append(result.get('response', ''))
                else:
                    response_texts.append('')  # Fallback for failed calls
                    
            return response_texts
            
        except Exception as e:
            print(f"Worker {self.worker_id}: LLM API error: {e}")
            # Return empty responses on failure
            return [''] * len(messages_list)
            
    def _step_environments(self, 
                          active_envs: List[WorkerEnvironment], 
                          llm_responses: List[str],
                          ctx_manager: ContextManager) -> List[WorkerEnvironment]:
        """Step environments with LLM responses."""
        remaining_active = []
        
        for env_wrapper, llm_response in zip(active_envs, llm_responses):
            if env_wrapper.status['terminated']:
                continue  # Skip already finished environments
                
            # Parse actions from LLM response
            actions = self._parse_actions_from_response(llm_response)
            
            # Execute actions in environment
            executed_actions = []
            acc_reward = 0.0
            turn_info = {}
            turn_done = False
            
            actions_left_before = env_wrapper.max_actions_per_traj - env_wrapper.status['num_actions']
            valid_actions = self._extract_valid_actions(env_wrapper, actions)
            
            # Check for action format penalty
            if len(valid_actions) != len(actions) or not valid_actions:
                env_wrapper.penalty += 0.1  # Format penalty
                
            # Execute actions
            for action in valid_actions[:actions_left_before]:
                try:
                    _, reward, done, info = env_wrapper.env.step(action)
                    acc_reward += reward
                    turn_info.update(info)
                    executed_actions.append(action)
                    
                    if done:
                        turn_done = True
                        break
                except Exception as e:
                    print(f"Worker {self.worker_id}: Environment step error: {e}")
                    break
                    
            # Update environment status
            env_wrapper.status['num_actions'] += len(executed_actions)
            env_wrapper.status['rewards'].append(acc_reward)
            
            if turn_done:
                env_wrapper.status['terminated'] = True
                env_wrapper.status['truncated'] = not turn_info.get('success', False)
            elif env_wrapper.status['num_actions'] >= env_wrapper.max_actions_per_traj:
                env_wrapper.status['truncated'] = True
                env_wrapper.status['terminated'] = True
                turn_done = True
                
            # Update history
            if env_wrapper.history:
                env_wrapper.history[-1].update({
                    'actions': executed_actions,
                    'reward': acc_reward,
                    'info': turn_info,
                    'llm_response': llm_response,
                    'llm_raw_response': llm_response
                })
                
            # Add new observation if not done
            if not turn_done:
                current_obs = self._handle_multimodal_state(env_wrapper.env.render())
                actions_left = env_wrapper.max_actions_per_traj - env_wrapper.status['num_actions']
                
                new_entry = {
                    'state': current_obs if isinstance(current_obs, str) else "<images>" * len(current_obs),
                    'actions_left': actions_left
                }
                
                if not isinstance(current_obs, str):
                    new_entry['images'] = current_obs
                    
                env_wrapper.history.append(new_entry)
                remaining_active.append(env_wrapper)
                
        return remaining_active
        
    def _parse_actions_from_response(self, response: str) -> List[str]:
        """Parse action commands from LLM response."""
        if not response:
            return []
            
        # Simple action parsing - can be enhanced based on your format
        action_sep = self.system_config.agent_proxy.get('action_sep', '||')
        
        # Look for action patterns
        import re
        
        # Try to find actions in various formats
        action_patterns = [
            r'Action:\s*(.+)',
            r'action:\s*(.+)',
            r'Actions:\s*(.+)',
            r'actions:\s*(.+)',
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                action_text = matches[0].strip()
                if action_sep in action_text:
                    actions = [a.strip() for a in action_text.split(action_sep)]
                else:
                    actions = [action_text]
                break
                
        # Fallback: use the entire response as a single action
        if not actions:
            actions = [response.strip()]
            
        return actions[:5]  # Limit number of actions
        
    def _extract_valid_actions(self, env_wrapper: WorkerEnvironment, actions: List[str]) -> List[str]:
        """Extract valid actions using environment's action lookup if available."""
        mapped_actions = []
        action_lookup = getattr(env_wrapper.config, 'action_lookup', None)
        
        if action_lookup is None:
            mapped_actions = actions
        else:
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions 
                            if action in rev_action_lookup]
                            
        return mapped_actions
        
    def _get_final_environment_state(self, env_wrapper: WorkerEnvironment) -> Dict[str, Any]:
        """Get final state for an environment."""
        import numpy as np
        
        # Compute environment metrics
        status = env_wrapper.status
        env_metric = {
            'success': float(status['terminated'] and (not status['truncated'])),
            'num_actions': status['num_actions'],
        }
        
        # Compute custom metrics from history
        custom_metric = {}
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']
        
        for turn in env_wrapper.history:
            for k, v in turn.get('info', {}).items():
                if k == 'success':
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))
                
        # Aggregate custom metrics
        for k, v in custom_metric.items():
            if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                env_metric[k] = np.sum(v) / max(1, len(env_wrapper.history) - 1)
            else:
                env_metric[k] = np.sum(v)
                
        # Add tag prefix to metrics
        env_metric = {f"{env_wrapper.tag}/{k}": v for k, v in env_metric.items()}
        
        # Create final state
        final_state = {
            "env_id": env_wrapper.env_id,
            "group_id": env_wrapper.group_id,  # Include group_id
            "history": env_wrapper.history,
            "tag": env_wrapper.tag,
            "penalty": env_wrapper.penalty,
            "metrics": env_metric,
            "status": status
        }
        
        # Add environment-specific information
        if env_wrapper.tag == "MetamathQA" and hasattr(env_wrapper.env, 'correct_answer'):
            final_state['correct_answer'] = env_wrapper.env.correct_answer
            
        # Add custom metrics to last history entry
        if env_wrapper.history:
            env_wrapper.history[-1]['metrics'] = custom_metric
            
        return final_state
        
    def _handle_multimodal_state(self, state):
        """Handle multimodal state (text or images)."""
        if isinstance(state, str):
            return state
        elif isinstance(state, list):
            # Import PIL here to avoid import issues in Ray workers
            try:
                import PIL.Image
                import numpy as np
                results = []
                for _state in state:
                    if isinstance(_state, np.ndarray):
                        results.append(PIL.Image.fromarray(_state, mode='RGB'))
                    else:
                        results.append(_state)
                return results
            except ImportError:
                return ["<image>"] * len(state)
        else:
            try:
                import PIL.Image
                import numpy as np
                if isinstance(state, np.ndarray):
                    return [PIL.Image.fromarray(state, mode='RGB')]
                else:
                    return [state]
            except ImportError:
                return "<image>"
                
    def _get_system_prompt(self, tag: str) -> str:
        """Get system prompt for environment type."""
        prompts = {
            "Alfworld": "You are an AI assistant helping to complete household tasks. Respond with specific actions to take.",
            "Sokoban": "You are playing Sokoban. Move the player and push boxes to goal positions. Use commands: up, down, left, right.",
            "Webshop": "You are shopping online. Navigate the website and find the requested items.",
            "MetamathQA": "Solve the given math problem step by step.",
            "Bandit": "Choose actions to maximize reward in this multi-armed bandit problem.",
            "FrozenLake": "Navigate the frozen lake to reach the goal while avoiding holes.",
            "Countdown": "Play the countdown numbers game to reach the target number.",
        }
        return prompts.get(tag, "You are an AI assistant. Follow the instructions carefully.")
        
    async def close(self):
        """Close all environments and clean up resources."""
        for env_wrapper in self.environments.values():
            if hasattr(env_wrapper.env, 'close'):
                env_wrapper.env.close()
                
        print(f"Worker {self.worker_id} closed {len(self.environments)} environments")