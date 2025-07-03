"""
Server Manager for VLLM API Server lifecycle management.
This module provides high-level management of VLLM API servers,
including startup, shutdown, health monitoring, and configuration.

Author: Generated for RAGEN independent rollout architecture
Date: 2025-06-25
"""

import os
import ray
import time
import asyncio
import threading
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .vllm_api_server import VLLMAPIServer


@dataclass
class ServerInfo:
    """Information about a running server."""
    server_id: str
    base_url: str
    config: Dict[str, Any]
    start_time: float
    host: str = "0.0.0.0"
    port: int = 8000
    is_healthy: bool = True
    error_count: int = 0
    last_health_check: float = field(default_factory=time.time)


class ServerManager:
    """
    Manager for VLLM API server lifecycle.
    
    Handles server startup, shutdown, health monitoring, and provides
    a clean interface for RAGEN rollout coordinators.
    """
    
    def __init__(self, auto_health_check: bool = True, health_check_interval: float = 30.0):
        """
        Initialize server manager.
        
        Args:
            auto_health_check: Whether to automatically monitor server health
            health_check_interval: Interval between health checks in seconds
        """
        self.servers: Dict[str, ServerInfo] = {}
        self.running_servers: Dict[str, VLLMAPIServer] = {}
        
        # Health monitoring
        self.auto_health_check = auto_health_check
        self.health_check_interval = health_check_interval
        self._health_monitor_task = None
        self._shutdown_event = threading.Event()
        
        # Ray initialization
        self._ensure_ray_initialized()
        
    def _ensure_ray_initialized(self):
        """Ensure Ray is initialized."""
        if not ray.is_initialized():
            # Use a minimal Ray setup for server management
            ray.init(
                ignore_reinit_error=True,
                configure_logging=False,
                log_to_driver=False
            )
            
    async def start_server(self, 
                          config: Dict[str, Any], 
                          server_id: str = "default",
                          host: str = "0.0.0.0", 
                          port: int = 8000,
                          wait_for_ready: bool = True) -> ServerInfo:
        """
        Start a VLLM API server.
        
        Args:
            config: RAGEN configuration for the server
            server_id: Unique identifier for this server
            host: Server host address
            port: Server port
            wait_for_ready: Whether to wait for server to be ready
            
        Returns:
            ServerInfo object with server details
            
        Raises:
            ValueError: If server_id already exists
            RuntimeError: If server fails to start
        """
        if server_id in self.servers:
            raise ValueError(f"Server with ID '{server_id}' already exists")
            
        print(f"Starting VLLM API server '{server_id}'...")
        
        try:
            # Create and start server
            server = VLLMAPIServer(config, host=host, port=port)
            base_url = await server.start()
            
            # Create server info
            server_info = ServerInfo(
                server_id=server_id,
                base_url=base_url,
                config=config,
                start_time=time.time(),
                host=host,
                port=port
            )
            
            # Store server references
            self.servers[server_id] = server_info
            self.running_servers[server_id] = server
            
            # Start health monitoring if this is the first server
            if self.auto_health_check and len(self.servers) == 1:
                await self._start_health_monitoring()
                
            print(f"✅ Server '{server_id}' started successfully at {base_url}")
            return server_info
            
        except Exception as e:
            print(f"❌ Failed to start server '{server_id}': {e}")
            # Cleanup on failure
            if server_id in self.servers:
                del self.servers[server_id]
            if server_id in self.running_servers:
                del self.running_servers[server_id]
            raise RuntimeError(f"Failed to start server '{server_id}': {e}")
            
    async def stop_server(self, server_id: str):
        """
        Stop a specific server.
        
        Args:
            server_id: ID of the server to stop
            
        Raises:
            ValueError: If server_id doesn't exist
        """
        if server_id not in self.servers:
            raise ValueError(f"No server found with ID '{server_id}'")
            
        print(f"Stopping server '{server_id}'...")
        
        try:
            # Stop the server
            server = self.running_servers[server_id]
            await server.stop()
            
            # Remove from tracking
            del self.servers[server_id]
            del self.running_servers[server_id]
            
            # Stop health monitoring if no servers remain
            if not self.servers and self._health_monitor_task:
                await self._stop_health_monitoring()
                
            print(f"✅ Server '{server_id}' stopped successfully")
            
        except Exception as e:
            print(f"⚠️  Error stopping server '{server_id}': {e}")
            # Still remove from tracking even if stop failed
            self.servers.pop(server_id, None)
            self.running_servers.pop(server_id, None)
            
    async def stop_all_servers(self):
        """Stop all running servers."""
        if not self.servers:
            print("No servers to stop")
            return
            
        print(f"Stopping {len(self.servers)} servers...")
        
        # Stop all servers concurrently
        stop_tasks = []
        for server_id in list(self.servers.keys()):
            stop_tasks.append(self.stop_server(server_id))
            
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Stop health monitoring
        if self._health_monitor_task:
            await self._stop_health_monitoring()
            
        print("✅ All servers stopped")
        
    def get_server(self, server_id: str = "default") -> Optional[ServerInfo]:
        """
        Get information about a server.
        
        Args:
            server_id: ID of the server
            
        Returns:
            ServerInfo if server exists, None otherwise
        """
        return self.servers.get(server_id)
        
    def list_servers(self) -> List[ServerInfo]:
        """Get information about all running servers."""
        return list(self.servers.values())
        
    def get_api_config(self, server_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get API configuration for a server.
        
        Args:
            server_id: ID of the server
            
        Returns:
            API configuration dictionary, or None if server doesn't exist
        """
        if server_id not in self.running_servers:
            return None
            
        server = self.running_servers[server_id]
        return server.get_api_config()
        
    async def health_check(self, server_id: str = "default") -> bool:
        """
        Check health of a specific server.
        
        Args:
            server_id: ID of the server to check
            
        Returns:
            True if server is healthy, False otherwise
        """
        if server_id not in self.running_servers:
            return False
            
        try:
            server = self.running_servers[server_id]
            is_healthy = await server.health_check()
            
            # Update server info
            if server_id in self.servers:
                self.servers[server_id].is_healthy = is_healthy
                self.servers[server_id].last_health_check = time.time()
                
                if not is_healthy:
                    self.servers[server_id].error_count += 1
                else:
                    self.servers[server_id].error_count = 0
                    
            return is_healthy
            
        except Exception as e:
            print(f"Health check failed for server '{server_id}': {e}")
            return False
            
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Check health of all servers.
        
        Returns:
            Dictionary mapping server_id to health status
        """
        health_results = {}
        
        health_tasks = []
        server_ids = list(self.servers.keys())
        
        for server_id in server_ids:
            health_tasks.append(self.health_check(server_id))
            
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        for server_id, result in zip(server_ids, results):
            if isinstance(result, bool):
                health_results[server_id] = result
            else:
                health_results[server_id] = False
                
        return health_results
        
    async def _start_health_monitoring(self):
        """Start background health monitoring."""
        if self._health_monitor_task is not None:
            return  # Already running
            
        print(f"Starting health monitoring (interval: {self.health_check_interval}s)")
        self._shutdown_event.clear()
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
    async def _stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._health_monitor_task is None:
            return
            
        print("Stopping health monitoring")
        self._shutdown_event.set()
        
        try:
            self._health_monitor_task.cancel()
            await self._health_monitor_task
        except asyncio.CancelledError:
            pass
            
        self._health_monitor_task = None
        
    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                if self.servers:
                    health_results = await self.health_check_all()
                    
                    # Log unhealthy servers
                    for server_id, is_healthy in health_results.items():
                        if not is_healthy:
                            server_info = self.servers.get(server_id)
                            if server_info:
                                print(f"⚠️  Server '{server_id}' is unhealthy (error count: {server_info.error_count})")
                                
                                # Consider restarting server if too many failures
                                if server_info.error_count >= 3:
                                    print(f"🔄 Considering restart for server '{server_id}' due to persistent failures")
                                    
                # Wait for next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
                
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop all servers."""
        if self.servers:
            # Run cleanup in a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            if loop.is_running():
                # If loop is already running, schedule cleanup
                asyncio.create_task(self.stop_all_servers())
            else:
                # Run cleanup directly
                loop.run_until_complete(self.stop_all_servers())


@asynccontextmanager
async def vllm_server_context(config: Dict[str, Any], **server_kwargs):
    """
    Async context manager for VLLM server lifecycle.
    
    Args:
        config: RAGEN configuration
        **server_kwargs: Additional server options
        
    Yields:
        ServerInfo for the started server
        
    Example:
        async with vllm_server_context(config, port=8001) as server_info:
            # Use server_info.base_url for API calls
            print(f"Server running at {server_info.base_url}")
    """
    manager = ServerManager(auto_health_check=False)
    
    try:
        server_info = await manager.start_server(config, **server_kwargs)
        yield server_info
    finally:
        await manager.stop_all_servers()


# Convenience functions
async def start_default_server(config: Dict[str, Any], **kwargs) -> ServerManager:
    """
    Start a default VLLM server with automatic management.
    
    Args:
        config: RAGEN configuration
        **kwargs: Additional server options
        
    Returns:
        ServerManager instance with running server
    """
    manager = ServerManager()
    await manager.start_server(config, **kwargs)
    return manager


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Example configuration
    example_config = {
        'system': {'CUDA_VISIBLE_DEVICES': "0"},
        'actor_rollout_ref': {
            'model': {'path': 'Qwen/Qwen2.5-7B-Instruct'},
            'rollout': {
                'tensor_model_parallel_size': 1,
                'max_model_len': 3600,
                'response_length': 400,
                'gpu_memory_utilization': 0.8,
            }
        }
    }
    
    async def test_manager():
        """Test server manager functionality."""
        print("Testing ServerManager...")
        
        async with vllm_server_context(example_config, server_id="test", port=8001) as server_info:
            print(f"✅ Server started: {server_info.base_url}")
            
            # Test health check
            manager = ServerManager(auto_health_check=False)
            healthy = await manager.health_check("test")
            print(f"Server healthy: {healthy}")
            
            # Get API config
            api_config = manager.get_api_config("test")
            print(f"API config: {api_config}")
            
            await asyncio.sleep(5)  # Keep running briefly
            
        print("✅ Server stopped via context manager")
        
    # Run test
    asyncio.run(test_manager())