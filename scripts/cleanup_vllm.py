#!/usr/bin/env python3
"""
Cleanup script to kill orphaned VLLM server processes.

Usage:
    python scripts/cleanup_vllm.py
    
This script will find and kill any running VLLM API server processes
that may have been left behind after interrupted executions.
"""

import subprocess
import sys
import os


def find_vllm_processes():
    """Find all running VLLM API server processes."""
    try:
        result = subprocess.run(
            ["ps", "-aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        vllm_processes = []
        for line in result.stdout.split('\n'):
            if 'vllm.entrypoints.openai.api_server' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    vllm_processes.append((pid, line.strip()))
        
        return vllm_processes
        
    except subprocess.CalledProcessError:
        print("Error running ps command")
        return []


def kill_vllm_processes(processes):
    """Kill the specified VLLM processes."""
    if not processes:
        print("No VLLM processes found.")
        return
        
    print(f"Found {len(processes)} VLLM process(es):")
    for pid, cmd_line in processes:
        print(f"  PID {pid}: {cmd_line[:100]}...")
        
    response = input("\nKill these processes? (y/N): ").lower()
    if response not in ['y', 'yes']:
        print("Aborted.")
        return
        
    killed_count = 0
    for pid, _ in processes:
        try:
            subprocess.run(["kill", "-9", pid], check=True)
            print(f"✅ Killed process {pid}")
            killed_count += 1
        except subprocess.CalledProcessError:
            print(f"❌ Failed to kill process {pid} (may already be dead)")
            
    print(f"\n✅ Killed {killed_count}/{len(processes)} processes")


def main():
    """Main cleanup function."""
    print("🔍 Searching for VLLM API server processes...")
    
    processes = find_vllm_processes()
    kill_vllm_processes(processes)
    
    # Double-check
    remaining = find_vllm_processes()
    if remaining:
        print(f"\n⚠️  {len(remaining)} processes still running:")
        for pid, cmd_line in remaining:
            print(f"  PID {pid}: {cmd_line[:100]}...")
        print("\nYou may need to kill these manually with: kill -9 <PID>")
    else:
        print("\n🎉 All VLLM processes cleaned up!")


if __name__ == "__main__":
    main() 