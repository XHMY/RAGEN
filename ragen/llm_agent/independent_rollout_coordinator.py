"""
Independent Rollout Coordinator for managing VLLM server + independent workers.
This module provides simplified coordination logic that manages server lifecycle
and independent environment workers.

Author: Generated for RAGEN independent rollout architecture
Date: 2025-06-25
"""

import ray
import asyncio
import random
from typing import Dict, Any, Optional, List
from collections import defaultdict

from .server_manager import ServerManager, vllm_server_context
from .independent_env_worker import IndependentEnvironmentWorker
from .ctx_manager import ContextManager
from ragen.env.alfworld_old.env import get_all_alfworld_game_files
from ragen.env import REGISTERED_ENV_CONFIGS
from verl import DataProto


class IndependentRolloutCoordinator:
    """
    Simplified coordinator for independent environment rollouts.
    
    This coordinator:
    - Manages VLLM API server lifecycle
    - Spawns independent environment workers
    - Collects and aggregates results
    - No LLM coordination needed (workers handle their own API calls)
    """
    
    def __init__(self, config: Dict[str, Any], tokenizer=None):
        """
        Initialize independent rollout coordinator.
        
        Args:
            config: System configuration
            tokenizer: Model tokenizer (optional, not used for API calls)
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # Server management
        self.server_manager = None
        self.server_info = None
        
        # Worker management
        self.env_workers = []
        
        # Ray initialization
        self._ensure_ray_initialized()
        
        print("IndependentRolloutCoordinator initialized")
        
    def _ensure_ray_initialized(self):
        """Ensure Ray is initialized."""
        if not ray.is_initialized():
            ray_config = self.config.get('ray_rollout', {}).get('ray_init_config', {})
            ray.init(ignore_reinit_error=True, **ray_config)
            print(f"Ray initialized with config: {ray_config}")
            
    async def start_vllm_server(self) -> str:
        """
        Start VLLM API server.
        
        Returns:
            Base URL of the started server
        """
        if self.server_info is not None:
            print("VLLM server already running")
            return self.server_info.base_url
            
        print("Starting VLLM API server...")
        
        # Create server manager
        self.server_manager = ServerManager(auto_health_check=True)
        
        # Get server configuration
        server_config = self.config.get('vllm_server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 8000)
        
        # Start server
        self.server_info = await self.server_manager.start_server(
            config=self.config,
            server_id="ragen_server",
            host=host,
            port=port,
            wait_for_ready=True
        )
        
        print(f"✅ VLLM server started at {self.server_info.base_url}")
        return self.server_info.base_url
        
    async def stop_vllm_server(self):
        """Stop VLLM API server."""
        if self.server_manager is not None:
            await self.server_manager.stop_all_servers()
            self.server_manager = None
            self.server_info = None
            print("✅ VLLM server stopped")
            
    def _create_environment_workers(self, mode: str = "train", num_workers: Optional[int] = None) -> List[ray.ObjectRef]:
        """
        Create independent environment workers.
        
        Args:
            mode: Either "train" or "val"
            num_workers: Number of workers to create (None = use config)
            
        Returns:
            List of Ray object references to environment workers
        """
        # Get environment configuration
        config = getattr(self.config.es_manager, mode)
        env_groups = int(config.env_groups)
        group_size = config.group_size
        total_envs = env_groups * group_size
        
        # Determine number of workers
        ray_config = self.config.get('ray_rollout', {})
        if num_workers is None:
            num_workers = ray_config.get('num_env_workers', 8)
        num_workers = min(num_workers, total_envs)
        
        print(f"Creating {num_workers} workers for {total_envs} environments")
        
        # Pre-load Alfworld game files if needed
        alfworld_game_files = None
        if 'Alfworld' in config.env_configs.tags:
            try:
                sample_config = REGISTERED_ENV_CONFIGS['alfworld_old']()
                alfworld_game_files = get_all_alfworld_game_files(sample_config.config_file)
                print(f"Pre-loaded {len(alfworld_game_files)} Alfworld game files")
            except Exception as e:
                print(f"Warning: Could not pre-load Alfworld game files: {e}")
                
        # Distribute environments across workers
        worker_specs = []
        env_idx = 0
        
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            cfg_template = self.config.custom_envs[tag]
            
            for group_id in range(n_group):
                for _ in range(group_size):
                    worker_id = env_idx % num_workers
                    
                    # Ensure worker spec exists
                    while len(worker_specs) <= worker_id:
                        worker_specs.append({
                            'worker_id': len(worker_specs),
                            'environments': []
                        })
                    
                    # Create environment spec
                    env_spec = {
                        'env_id': env_idx,
                        'tag': tag,
                        'env_config': {
                            'env_type': cfg_template.env_type,
                            'env_config': cfg_template.env_config
                        },
                        'max_actions_per_traj': cfg_template.max_actions_per_traj,
                        'group_id': group_id
                    }
                    
                    # Assign Alfworld game file if available
                    if tag == 'Alfworld' and alfworld_game_files:
                        game_file_index = env_idx % len(alfworld_game_files)
                        env_spec['alfworld_game_file'] = alfworld_game_files[game_file_index]
                    
                    worker_specs[worker_id]['environments'].append(env_spec)
                    env_idx += 1
                    
        # Create LLM configuration for workers
        if self.server_info is None:
            raise RuntimeError("VLLM server must be started before creating workers")
            
        llm_config = self.server_manager.get_api_config("ragen_server")
        
        # Create Ray actors for independent workers
        env_workers = []
        for spec in worker_specs:
            if spec['environments']:  # Only create workers with environments
                worker = IndependentEnvironmentWorker.remote(
                    worker_id=spec['worker_id'],
                    environment_specs=spec['environments'],
                    llm_config=llm_config,
                    system_config=self.config
                )
                env_workers.append(worker)
                
        print(f"Created {len(env_workers)} independent environment workers")
        return env_workers
        
    async def independent_rollout(self, 
                                 dataproto: DataProto, 
                                 val: bool = False,
                                 num_workers: Optional[int] = None,
                                 auto_manage_server: bool = True) -> DataProto:
        """
        Perform independent rollout using VLLM server + independent workers.
        
        Args:
            dataproto: Input data protocol with meta information
            val: Whether to use validation mode
            num_workers: Number of environment workers (None = use config)
            auto_manage_server: Whether to automatically start/stop server
            
        Returns:
            DataProto containing rollout results
        """
        mode = "val" if val else "train"
        max_turns = self.config.agent_proxy.max_turn
        
        print(f"Starting independent rollout (mode={mode}, workers={num_workers}, max_turns={max_turns})")
        
        try:
            # Start VLLM server if needed
            if auto_manage_server and self.server_info is None:
                await self.start_vllm_server()
                
            # Create environment workers
            env_workers = self._create_environment_workers(mode, num_workers)
            self.env_workers = env_workers
            
            # Run independent rollouts in parallel
            print(f"Running rollouts on {len(env_workers)} workers...")
            
            rollout_tasks = []
            for worker in env_workers:
                seed = random.randint(0, 1000000) if mode == "train" else 123
                task = worker.run_independent_rollout.remote(
                    seed=seed,
                    mode=mode,
                    max_turns=max_turns
                )
                rollout_tasks.append(task)
                
            # Wait for all rollouts to complete
            worker_results = await asyncio.gather(*[
                self._ray_task_to_asyncio(task) for task in rollout_tasks
            ])
            
            # Aggregate results from all workers
            all_final_states = []
            for worker_result in worker_results:
                if worker_result:
                    all_final_states.extend(worker_result)
                    
            print(f"Collected {len(all_final_states)} final environment states")
            
            # Convert to rollouts using context manager
            ctx_manager = ContextManager(self.config, self.tokenizer, mode=mode)
            rollouts = ctx_manager.formulate_rollouts(all_final_states)
            
            return rollouts
            
        finally:
            # Clean up workers
            if self.env_workers:
                cleanup_tasks = [worker.close.remote() for worker in self.env_workers]
                try:
                    await asyncio.gather(*[
                        self._ray_task_to_asyncio(task) for task in cleanup_tasks
                    ], return_exceptions=True)
                except Exception as e:
                    print(f"Warning: Error during worker cleanup: {e}")
                self.env_workers = []
                
            # Stop server if auto-managed
            if auto_manage_server:
                await self.stop_vllm_server()
                
    async def _ray_task_to_asyncio(self, ray_task):
        """Convert Ray task to asyncio-compatible awaitable."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, ray_task)
        
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of VLLM server."""
        if self.server_info is None:
            return {"status": "not_running"}
            
        return {
            "status": "running",
            "base_url": self.server_info.base_url,
            "start_time": self.server_info.start_time,
            "is_healthy": self.server_info.is_healthy
        }
        
    async def health_check(self) -> bool:
        """Check health of VLLM server."""
        if self.server_manager is None:
            return False
        return await self.server_manager.health_check("ragen_server")
        
    async def shutdown(self):
        """Shutdown coordinator and clean up all resources."""
        print("Shutting down IndependentRolloutCoordinator...")
        
        # Clean up workers
        if self.env_workers:
            cleanup_tasks = [worker.close.remote() for worker in self.env_workers]
            try:
                await asyncio.gather(*[
                    self._ray_task_to_asyncio(task) for task in cleanup_tasks
                ], return_exceptions=True)
            except Exception as e:
                print(f"Warning: Error during worker cleanup: {e}")
            self.env_workers = []
            
        # Stop server
        await self.stop_vllm_server()
        
        print("✅ IndependentRolloutCoordinator shutdown completed")


# Context manager for easy server + rollout management
class IndependentRolloutContext:
    """Context manager for independent rollout with automatic server management."""
    
    def __init__(self, config: Dict[str, Any], tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.coordinator = None
        
    async def __aenter__(self):
        self.coordinator = IndependentRolloutCoordinator(self.config, self.tokenizer)
        await self.coordinator.start_vllm_server()
        return self.coordinator
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.coordinator:
            await self.coordinator.shutdown()


# Convenience functions
async def run_independent_rollout(config: Dict[str, Any], 
                                 dataproto: DataProto,
                                 tokenizer=None,
                                 **kwargs) -> DataProto:
    """
    Run independent rollout with automatic server management.
    
    Args:
        config: System configuration
        dataproto: Input data protocol
        tokenizer: Model tokenizer (optional)
        **kwargs: Additional rollout options
        
    Returns:
        DataProto containing rollout results
    """
    async with IndependentRolloutContext(config, tokenizer) as coordinator:
        return await coordinator.independent_rollout(dataproto, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from omegaconf import OmegaConf
    from verl import DataProto
    
    # Example configuration
    example_config = {
        'system': {'CUDA_VISIBLE_DEVICES': "0,1"},
        'actor_rollout_ref': {
            'model': {'path': 'Qwen/Qwen2.5-7B-Instruct'},
            'rollout': {
                'tensor_model_parallel_size': 2,
                'max_model_len': 3600,
                'response_length': 400,
                'gpu_memory_utilization': 0.8,
                'temperature': 0.5,
            }
        },
        'agent_proxy': {
            'max_turn': 5,
            'action_sep': '||'
        },
        'es_manager': {
            'train': {
                'env_groups': 2,
                'group_size': 4,
                'env_configs': {
                    'tags': ['Sokoban'],
                    'n_groups': [2]
                }
            }
        },
        'custom_envs': {
            'Sokoban': {
                'env_type': 'sokoban',
                'max_actions_per_traj': 50,
                'env_config': None
            }
        },
        'ray_rollout': {
            'num_env_workers': 4,
            'ray_init_config': {
                'num_cpus': 6,
                'num_gpus': 2
            }
        },
        'vllm_server': {
            'host': '0.0.0.0',
            'port': 8001,
            'accelerator_type': 'A100'
        }
    }
    
    async def test_independent_rollout():
        """Test independent rollout functionality."""
        print("Testing IndependentRolloutCoordinator...")
        
        config_obj = OmegaConf.create(example_config)
        
        # Create test dataproto
        test_dataproto = DataProto(
            batch=None,
            non_tensor_batch=None,
            meta_info={
                'eos_token_id': 151645,
                'pad_token_id': 151643,
                'recompute_log_prob': False,
                'do_sample': True,
                'validate': True
            }
        )
        
        try:
            # Test with context manager
            async with IndependentRolloutContext(config_obj) as coordinator:
                print(f"✅ Server started: {coordinator.get_server_status()}")
                
                # Test health check
                healthy = await coordinator.health_check()
                print(f"Server healthy: {healthy}")
                
                # Run rollout
                print("Running independent rollout...")
                rollouts = await coordinator.independent_rollout(
                    test_dataproto, 
                    val=True, 
                    num_workers=2,
                    auto_manage_server=False  # Already managed by context
                )
                
                print(f"✅ Rollout completed!")
                print(f"Results: {len(rollouts.batch['rm_scores'])} trajectories")
                
                # Calculate metrics
                rewards = rollouts.batch["rm_scores"].sum(-1).mean().item()
                metrics = rollouts.meta_info.get("metrics", {})
                
                print(f"Average reward: {rewards:.4f}")
                for metric_name, metric_value in metrics.items():
                    print(f"  {metric_name}: {metric_value:.4f}")
                    
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
        print("Test completed!")
        
    # Run test
    asyncio.run(test_independent_rollout())