"""
Parallel Agent Proxy using Ray for distributed environment rollouts.
This module extends the original LLMAgentProxy to support Ray-based parallel processing.

Author: Generated for RAGEN parallel rollout
Date: 2025-06-25
"""

import os
import time
import hydra
from typing import Optional
from transformers import AutoTokenizer

from verl import DataProto
from .agent_proxy import LLMAgentProxy, VllmWrapperWg, ApiCallingWrapperWg
from .independent_rollout_coordinator import IndependentRolloutCoordinator, IndependentRolloutContext
from .ctx_manager import ContextManager
from .es_manager import EnvStateManager


class ParallelLLMAgentProxy(LLMAgentProxy):
    """
    Extended LLM Agent Proxy with API-based parallel rollout support.
    
    This class extends the original LLMAgentProxy to provide both sequential
    and independent parallel rollout capabilities using VLLM API server,
    allowing for significant speedup on slow environments like Alfworld.
    """
    
    def __init__(self, config, actor_rollout_wg, tokenizer, use_ray: bool = None):
        """
        Initialize the parallel agent proxy.
        
        Args:
            config: System configuration
            actor_rollout_wg: Actor worker group for LLM inference (can be None for API-only mode)
            tokenizer: Model tokenizer
            use_ray: Whether to use Ray for parallel rollouts (None = auto-detect from config)
        """
        # Initialize parent class - handle None actor_rollout_wg for API-only mode
        if actor_rollout_wg is not None:
            super().__init__(config, actor_rollout_wg, tokenizer)
        else:
            # For API-only mode, initialize minimal state
            self.config = config
            self.tokenizer = tokenizer
            self.actor_wg = None  # No actor worker group needed for API calls
        
        # Determine whether to use Ray
        if use_ray is None:
            use_ray = getattr(config, 'ray_rollout', {}).get('use_ray', False)
        self.use_ray = use_ray
        
        # Store config for context manager
        self.rollout_context = None
        
        if self.use_ray:
            print(f"ParallelLLMAgentProxy initialized with independent rollout mode (API-only)")
        else:
            print("ParallelLLMAgentProxy initialized in sequential mode")
    
    def rollout(self, dataproto: DataProto, val: bool = False, 
                force_sequential: bool = False, num_workers: Optional[int] = None) -> DataProto:
        """
        Perform rollout with automatic selection between different modes.
        
        Args:
            dataproto: Input data protocol
            val: Whether to use validation mode
            force_sequential: Force sequential rollout even if Ray is available
            num_workers: Override number of environment workers for this rollout
            
        Returns:
            DataProto containing rollout results
        """
        if force_sequential or not self.use_ray:
            return self._sequential_rollout(dataproto, val)
        else:
            # Use independent rollout with context manager (API-only)
            return self._independent_rollout(dataproto, val, num_workers)
    
    def _independent_rollout(self, dataproto: DataProto, val: bool = False, 
                            num_workers: Optional[int] = None) -> DataProto:
        """
        Perform independent rollout using VLLM server + independent workers.
        Uses the context manager pattern like in the example to ensure proper server management.
        
        Args:
            dataproto: Input data protocol
            val: Whether to use validation mode
            num_workers: Number of environment workers to use
            
        Returns:
            DataProto containing rollout results
        """
        print(f"🚀 Starting independent rollout (val={val}, num_workers={num_workers})")
        start_time = time.time()
        
        # Use the context manager pattern like in the example
        import asyncio
        
        async def run_rollout():
            # Use IndependentRolloutContext for automatic server management
            async with IndependentRolloutContext(self.config, self.tokenizer) as coordinator:
                print(f"✅ VLLM server started: {coordinator.get_server_status()['base_url']}")
                
                # Run independent rollout with automatic server management
                rollouts = await coordinator.independent_rollout(
                    dataproto, 
                    val=val, 
                    num_workers=num_workers,
                    auto_manage_server=False  # Already managed by context
                )
                
                return rollouts
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a task
                import nest_asyncio
                nest_asyncio.apply()
                rollouts = loop.run_until_complete(run_rollout())
            else:
                rollouts = loop.run_until_complete(run_rollout())
        except RuntimeError:
            # No event loop, create new one
            rollouts = asyncio.run(run_rollout())
        
        end_time = time.time()
        rollout_time = end_time - start_time
        
        # Add timing information to meta_info
        if rollouts.meta_info is None:
            rollouts.meta_info = {}
        rollouts.meta_info["rollout_time"] = rollout_time
        rollouts.meta_info["rollout_mode"] = "independent_api_only"
        
        print(f"✅ Independent rollout completed in {rollout_time:.2f} seconds")
        return rollouts
        
    
    def _sequential_rollout(self, dataproto: DataProto, val: bool = False) -> DataProto:
        """
        Perform sequential rollout using the original implementation.
        
        Args:
            dataproto: Input data protocol
            val: Whether to use validation mode
            
        Returns:
            DataProto containing rollout results
        """
        print(f"Starting sequential rollout (val={val})")
        start_time = time.time()
        
        # Use parent class implementation
        rollouts = super().rollout(dataproto, val)
        
        end_time = time.time()
        print(f"Sequential rollout completed in {end_time - start_time:.2f} seconds")
        
        return rollouts
    
    def compare_rollout_methods(self, dataproto: DataProto, val: bool = False, 
                               num_workers: Optional[int] = None) -> dict:
        """
        Compare performance between sequential and parallel rollout methods.
        
        Args:
            dataproto: Input data protocol
            val: Whether to use validation mode
            num_workers: Number of workers for parallel rollout
            
        Returns:
            Dictionary containing comparison results
        """
        results = {}
        
        # Sequential rollout
        print("=== Sequential Rollout ===")
        seq_start = time.time()
        seq_rollouts = self._sequential_rollout(dataproto, val)
        seq_time = time.time() - seq_start
        
        # Calculate sequential metrics
        seq_rewards = seq_rollouts.batch["rm_scores"].sum(-1).mean().item()
        seq_metrics = seq_rollouts.meta_info["metrics"]
        
        results['sequential'] = {
            'time': seq_time,
            'avg_reward': seq_rewards,
            'metrics': seq_metrics,
            'rollouts': seq_rollouts
        }
        
        # Independent parallel rollout
        if self.use_ray and self.independent_coordinator is not None:
            print("=== Independent Parallel Rollout ===")
            par_start = time.time()
            par_rollouts = self._independent_rollout(dataproto, val, num_workers)
            par_time = time.time() - par_start
            
            # Calculate parallel metrics
            par_rewards = par_rollouts.batch["rm_scores"].sum(-1).mean().item()
            par_metrics = par_rollouts.meta_info["metrics"]
            
            results['independent'] = {
                'time': par_time,
                'avg_reward': par_rewards,
                'metrics': par_metrics,
                'rollouts': par_rollouts
            }
            
            # Calculate speedup
            speedup = seq_time / par_time if par_time > 0 else float('inf')
            results['speedup'] = speedup
            
            print(f"=== Comparison Results ===")
            print(f"Sequential time: {seq_time:.2f}s")
            print(f"Independent time: {par_time:.2f}s")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Sequential reward: {seq_rewards:.4f}")
            print(f"Independent reward: {par_rewards:.4f}")
        else:
            print("Independent rollout not available for comparison")
            
        return results
    
    def get_rollout_stats(self) -> dict:
        """
        Get statistics about rollout configuration and capabilities.
        
        Returns:
            Dictionary containing rollout statistics
        """
        stats = {
            'use_ray': self.use_ray,
            'has_independent_coordinator': self.independent_coordinator is not None,
            'actor_type': type(self.actor_wg).__name__,
        }
        
        if self.independent_coordinator:
            stats.update({
                'rollout_mode': 'independent',
                'server_status': self.independent_coordinator.get_server_status(),
            })
            
        return stats
    
    def shutdown(self):
        """Shutdown parallel components and clean up resources."""
        if self.independent_coordinator:
            import asyncio
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule shutdown task
                    asyncio.create_task(self.independent_coordinator.shutdown())
                else:
                    loop.run_until_complete(self.independent_coordinator.shutdown())
            except RuntimeError:
                # No event loop, create new one
                asyncio.run(self.independent_coordinator.shutdown())
            print("Independent coordinator shutdown completed")


# Compatibility functions for easy migration
def create_parallel_agent_proxy(config, tokenizer, use_vllm: bool = None, 
                               use_ray: bool = None) -> ParallelLLMAgentProxy:
    """
    Convenience function to create a ParallelLLMAgentProxy.
    For independent rollouts, uses pure API approach with no model loading.
    
    Args:
        config: System configuration
        tokenizer: Model tokenizer
        use_vllm: Whether to use VLLM (True) or API calling (False). If None, auto-detect from config.
        use_ray: Whether to enable Ray-based parallel rollouts
        
    Returns:
        Configured ParallelLLMAgentProxy instance
    """
    # Auto-detect the appropriate mode based on config
    ray_config = getattr(config, 'ray_rollout', {})
    rollout_mode = ray_config.get('rollout_mode', 'sequential')
    
    # For independent rollout mode, ALWAYS use API-only approach (no model loading)
    if rollout_mode == 'independent':
        print("🚀 Using API-only mode for independent rollout (no model loading)")
        # Pass None as actor_wg for pure API approach
        proxy = ParallelLLMAgentProxy(config, actor_rollout_wg=None, tokenizer=tokenizer, use_ray=use_ray)
    else:
        print("📦 Using VLLM worker for sequential rollout mode")
        # Create VLLM worker for sequential mode
        actor_wg = VllmWrapperWg(config, tokenizer)
        proxy = ParallelLLMAgentProxy(config, actor_wg, tokenizer, use_ray=use_ray)
    
    return proxy


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Main function for testing ParallelLLMAgentProxy.
    """
    # Set up environment
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    
    # Initialize tokenizer and proxy
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    
    # Test with different configurations
    print("=== Testing Sequential Mode ===")
    proxy_seq = create_parallel_agent_proxy(config, tokenizer, use_ray=False)
    
    # Create test dataproto
    test_dataproto = DataProto(
        batch=None, 
        non_tensor_batch=None, 
        meta_info={
            'eos_token_id': 151645, 
            'pad_token_id': 151643, 
            'recompute_log_prob': False, 
            'do_sample': config.actor_rollout_ref.rollout.do_sample, 
            'validate': True
        }
    )
    
    # Run sequential rollout
    seq_rollouts = proxy_seq.rollout(test_dataproto, val=True)
    seq_rewards = seq_rollouts.batch["rm_scores"].sum(-1).mean().item()
    print(f"Sequential rollout reward: {seq_rewards:.4f}")
    
    # Test with Ray if configured
    ray_config = getattr(config, 'ray_rollout', {})
    if ray_config.get('use_ray', False):
        print("\n=== Testing Parallel Mode ===")
        proxy_par = create_parallel_agent_proxy(config, tokenizer, use_ray=True)
        
        # Compare rollout methods
        comparison = proxy_par.compare_rollout_methods(test_dataproto, val=True)
        
        # Print detailed results
        for method, results in comparison.items():
            if method != 'speedup':
                print(f"\n{method.title()} Results:")
                print(f"  Time: {results['time']:.2f}s")
                print(f"  Avg Reward: {results['avg_reward']:.4f}")
                for metric_name, metric_value in results['metrics'].items():
                    print(f"  {metric_name}: {metric_value:.4f}")
        
        if 'speedup' in comparison:
            print(f"\nSpeedup: {comparison['speedup']:.2f}x")
            
        proxy_par.shutdown()
    else:
        print("\nRay rollout not configured. Set ray_rollout.use_ray=true in config to test parallel mode.")
    
    print("\nTesting completed!")


if __name__ == "__main__":
    main()