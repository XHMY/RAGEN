"""
VLLM API Server using VLLM's native API server for centralized LLM inference.
This module provides an OpenAI-compatible API server backed by VLLM,
allowing multiple Ray workers to make independent API calls.

Author: Generated for RAGEN independent rollout architecture
Date: 2025-06-25
"""

import os
import asyncio
import subprocess
import signal
import time
import atexit
import weakref
from typing import Dict, Any, Optional
import aiohttp

# Global tracking of server instances for cleanup
_server_instances = weakref.WeakSet()

def _cleanup_all_servers():
    """Clean up all running VLLM servers."""
    print("Cleaning up VLLM servers...")
    for server in list(_server_instances):
        try:
            if hasattr(server, 'stop_sync'):
                server.stop_sync()
        except Exception as e:
            print(f"Error cleaning up server: {e}")

def _signal_handler(signum, frame):
    """Handle interrupt signals."""
    print(f"\nReceived signal {signum}, cleaning up...")
    _cleanup_all_servers()
    exit(0)

# Register cleanup functions
atexit.register(_cleanup_all_servers)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class VLLMAPIServer:
    """
    VLLM API Server manager using VLLM's native OpenAI-compatible server.
    Provides OpenAI-compatible API endpoints for RAGEN rollouts.
    """
    
    def __init__(self, config: Dict[str, Any], host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize VLLM API server.
        
        Args:
            config: RAGEN configuration
            host: Server host address
            port: Server port
        """
        self.config = config
        self.host = host
        self.port = port
        self.process = None
        self.is_running = False
        
        # Extract model and rollout configuration
        self.model_path = config.get('actor_rollout_ref', {}).get('model', {}).get('path', 'Qwen/Qwen2.5-7B-Instruct')
        self.rollout_config = config.get('actor_rollout_ref', {}).get('rollout', {})
        
        # Extract server configuration
        server_config = config.get('vllm_server', {})
        self.route_prefix = server_config.get('route_prefix', '/v1')
        
        # Register for cleanup
        _server_instances.add(self)
        
    async def start(self) -> str:
        """
        Start the VLLM API server.
        
        Returns:
            Base URL of the started server
        """
        if self.is_running:
            print("VLLM API server is already running")
            return self._get_base_url()
            
        print("Starting VLLM API server...")
        
        # Build VLLM server command
        cmd = self._build_vllm_command()
        
        try:
            # Start VLLM server process
            print(f"Running command: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Wait for server to be ready
            await self._wait_for_server()
            
            self.is_running = True
            base_url = self._get_base_url()
            print(f"✅ VLLM API server started at {base_url}")
            return base_url
            
        except Exception as e:
            print(f"❌ Failed to start VLLM API server: {e}")
            if self.process:
                self.process.terminate()
                self.process = None
            raise
            
    async def stop(self):
        """Stop the VLLM API server."""
        if not self.is_running or not self.process:
            print("VLLM API server is not running")
            return
            
        print("Stopping VLLM API server...")
        
        try:
            pid = self.process.pid
            
            # First try gentle termination
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                print("Process already terminated")
                self.process = None
                self.is_running = False
                return
            except OSError:
                # Fallback to direct process termination
                self.process.terminate()
            
            # Wait for process to terminate
            for _ in range(10):  # Wait up to 10 seconds
                if self.process.poll() is not None:
                    break
                await asyncio.sleep(1)
            
            # Force kill if still running
            if self.process.poll() is None:
                print("Force killing VLLM server...")
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    # Process group doesn't exist, try direct kill
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Process is already dead
                
                # Wait a bit more for forced termination
                for _ in range(5):
                    if self.process.poll() is not None:
                        break
                    await asyncio.sleep(0.5)
                
            self.process = None
            self.is_running = False
            print("✅ VLLM API server stopped")
            
        except Exception as e:
            print(f"⚠️  Error stopping VLLM API server: {e}")
            # Clean up state even if stop failed
            self.process = None
            self.is_running = False
    
    def stop_sync(self):
        """Synchronous version of stop for use in signal handlers and destructors."""
        if not self.is_running or not self.process:
            return
            
        print("Stopping VLLM API server (sync)...")
        
        try:
            pid = self.process.pid
            
            # First try gentle termination
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                try:
                    self.process.terminate()
                except:
                    pass
            
            # Wait for process to terminate
            for _ in range(5):  # Wait up to 5 seconds for sync version
                if self.process.poll() is not None:
                    break
                time.sleep(1)
            
            # Force kill if still running
            if self.process.poll() is None:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                
                # Final wait
                for _ in range(3):
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.5)
                
            self.process = None
            self.is_running = False
            print("✅ VLLM API server stopped (sync)")
            
        except Exception as e:
            print(f"⚠️  Error stopping VLLM API server (sync): {e}")
            self.process = None
            self.is_running = False
    
    def __del__(self):
        """Destructor to ensure process cleanup."""
        if self.is_running and self.process:
            self.stop_sync()
            
    def _build_vllm_command(self) -> list:
        """Build the VLLM server command."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
        ]
        
        # Add tensor parallel size
        tensor_parallel_size = self.rollout_config.get('tensor_model_parallel_size', 1)
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        
        # Add GPU memory utilization
        gpu_memory_utilization = self.rollout_config.get('gpu_memory_utilization', 0.8)
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        
        # Add max model length
        max_model_len = self.rollout_config.get('max_model_len', 3600)
        cmd.extend(["--max-model-len", str(max_model_len)])
        
        # Add max num seqs
        max_num_seqs = self.rollout_config.get('max_num_seqs', 32)
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
        
        # Add other VLLM arguments
        if self.rollout_config.get('enable_chunked_prefill', True):
            cmd.append("--enable-chunked-prefill")
            
        if self.rollout_config.get('enable_prefix_caching', True):
            cmd.append("--enable-prefix-caching")
            
        if self.rollout_config.get('enforce_eager', False):
            cmd.append("--enforce-eager")
            
        if self.rollout_config.get('disable_log_stats', True):
            cmd.append("--disable-log-stats")
            
        # Add max num batched tokens
        max_num_batched_tokens = self.rollout_config.get('max_num_batched_tokens', 8192)
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
        
        # Add trust remote code
        cmd.append("--trust-remote-code")
        
        # Add dtype
        dtype = self.rollout_config.get('dtype', 'auto')
        cmd.extend(["--dtype", dtype])
        
        return cmd
        
    def _get_base_url(self) -> str:
        """Get the base URL of the server."""
        # Use localhost for client connections when server binds to 0.0.0.0
        client_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{client_host}:{self.port}/v1"
        
    async def _wait_for_server(self, timeout: int = 120):
        """
        Wait for the server to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        base_url = self._get_base_url()
        health_url = f"{base_url}/models"  # OpenAI models endpoint
        
        print(f"Waiting for server to be ready at {health_url}...")
        
        for attempt in range(timeout):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=5) as response:
                        if response.status == 200:
                            print(f"✅ Server is ready after {attempt + 1} seconds")
                            return
            except Exception:
                pass  # Server not ready yet
                
            await asyncio.sleep(1)
            
            # Check if process died
            if self.process and self.process.poll() is not None:
                # Process died, get error output
                stdout, stderr = self.process.communicate()
                error_msg = f"VLLM server process died. STDERR: {stderr.decode()[:500]}"
                raise RuntimeError(error_msg)
                
        raise RuntimeError(f"Server failed to start within {timeout} seconds")
        
    async def health_check(self) -> bool:
        """
        Check if the server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        if not self.is_running:
            return False
            
        try:
            base_url = self._get_base_url()
            health_url = f"{base_url}/models"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False
            
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the server.
        
        Returns:
            Dictionary containing server information
        """
        return {
            "base_url": self._get_base_url() if self.is_running else None,
            "host": self.host,
            "port": self.port,
            "model_path": self.model_path,
            "is_running": self.is_running,
            "process_id": self.process.pid if self.process else None,
        }
        
    def get_api_config(self) -> Dict[str, Any]:
        """
        Get API configuration for LLM clients.
        
        Returns:
            Dictionary containing API configuration for connecting to this server
        """
        return {
            "provider_name": "vllm_local",
            "model_name": self.model_path,  # Use actual model path as VLLM exposes it
            "base_url": self._get_base_url() if self.is_running else None,
            "api_key": "dummy_key",  # VLLM doesn't require real API keys
            "generation_kwargs": {
                "temperature": self.rollout_config.get("temperature", 0.7),
                "max_tokens": self.rollout_config.get("response_length", 500),
                "top_p": self.rollout_config.get("top_p", 0.9),
                # Note: VLLM OpenAI API doesn't support top_k, only top_p
            }
        }


# Convenience function for creating and managing VLLM servers
async def create_vllm_server(config: Dict[str, Any], host: str = "0.0.0.0", 
                           port: int = 8000) -> VLLMAPIServer:
    """
    Create and start a VLLM API server.
    
    Args:
        config: RAGEN configuration
        host: Server host
        port: Server port
        
    Returns:
        Started VLLMAPIServer instance
    """
    server = VLLMAPIServer(config, host, port)
    await server.start()
    return server


if __name__ == "__main__":
    # Test the VLLM API server
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Minimal test configuration
    test_config = {
        "actor_rollout_ref": {
            "model": {
                "path": "microsoft/DialoGPT-medium"  # Smaller model for testing
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 1024,
                "max_num_seqs": 8,
            }
        }
    }
    
    async def test_server():
        print("Testing VLLM API server...")
        server = VLLMAPIServer(test_config, port=8001)
        
        try:
            await server.start()
            
            # Test health check
            healthy = await server.health_check()
            print(f"Server healthy: {healthy}")
            
            # Keep server running briefly
            await asyncio.sleep(5)
            
        finally:
            await server.stop()
            
    asyncio.run(test_server())