# Parallel Environment Rollout with Joblib

This document describes the new parallel environment rollout feature implemented using joblib for the RAGEN framework.

## Overview

The parallel rollout system allows multiple environment instances to be sampled concurrently using joblib workers, potentially improving the speed of environment sampling during RL training. Each worker creates and manages its own subset of environments, while sharing access to the LLM inference engine.

## Key Features

- **Joblib-based Parallelization**: Uses joblib for process-based or thread-based parallelism
- **Dynamic Environment Creation**: Environments are created within worker processes, eliminating the need to maintain persistent environment instances
- **Shared LLM Inference**: All workers share access to the same LLM inference engine in a thread-safe manner
- **Fallback Support**: Automatically falls back to sequential execution if parallel execution fails
- **Configurable Worker Count**: Supports configurable number of parallel workers

## Architecture

```
Main Process (Agent Trainer)
├── ParallelRolloutManager
    ├── SharedLLMInferenceManager (thread-safe LLM access)
    └── JobLib Workers (1 to N)
        ├── Worker 1 (handles envs 0-15)
        │   ├── Creates local environments
        │   ├── Runs rollout loop
        │   └── Returns rollout data
        ├── Worker 2 (handles envs 16-31)
        └── ...
```

## Usage

### 1. Configuration

Enable parallel rollout in your config file:

```yaml
agent_proxy:
  use_parallel_rollout: True   # Enable parallel rollout
  parallel_n_jobs: 4          # Number of workers (-1 for auto)
  # ... other config options
```

### 2. Using Pre-configured Settings

Use the provided parallel configuration:

```bash
python train.py --config-name base-parallel
```

### 3. Testing the Implementation

Run the test script to compare performance:

```bash
python test_parallel_rollout.py --config-name base-parallel
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_parallel_rollout` | Enable/disable parallel rollout | `False` |
| `parallel_n_jobs` | Number of parallel workers (-1 for auto) | `-1` |

## Performance Considerations

### When to Use Parallel Rollout

- **Large number of environments**: More environments mean better parallelization opportunities
- **CPU-bound environment logic**: Environments with significant computation benefit more
- **Multi-core systems**: Systems with multiple CPU cores see better improvements

### When Sequential Might Be Better

- **Small number of environments**: Overhead might outweigh benefits
- **GPU-memory constrained**: Parallel workers might compete for GPU memory
- **Simple environments**: Very fast environments might not benefit from parallelization

## Implementation Details

### Environment Management

- Each worker creates its own environment instances locally
- No persistent environment state between rollouts
- Environments are properly closed after each rollout

### LLM Inference Coordination

- Thread-safe shared LLM inference manager
- Workers coordinate access to prevent conflicts
- Supports both Ray worker groups and VLLM wrappers

### Error Handling

- Automatic fallback to sequential execution on errors
- Comprehensive error logging and debugging
- Graceful handling of worker failures

## Compatibility

The parallel rollout system is designed to be backward-compatible:

- Existing configs work unchanged (parallel rollout disabled by default)
- All environment types are supported
- Works with existing reward functions and metrics

## Debugging

### Common Issues

1. **Memory Issues**: Reduce `parallel_n_jobs` if running out of memory
2. **Threading Conflicts**: Try different joblib backends (`threading`, `multiprocessing`)
3. **Environment Initialization**: Ensure all environments can be created in worker processes

### Debug Mode

For debugging, you can disable parallel rollout temporarily:

```yaml
agent_proxy:
  use_parallel_rollout: False
```

### Logging

The system provides detailed logging:
- Worker initialization and assignment
- Execution timing information
- Error messages and fallback notifications

## Future Improvements

- [ ] Support for distributed rollout across multiple nodes
- [ ] Adaptive worker count based on system load
- [ ] Environment instance pooling for reuse
- [ ] Advanced load balancing strategies

## Testing

The `test_parallel_rollout.py` script provides comprehensive testing:

- Performance comparison between sequential and parallel
- Result consistency verification
- Error handling validation
- Metric comparison

Run tests with:
```bash
python test_parallel_rollout.py --config-name base-parallel
```

## Contributing

When modifying the parallel rollout system:

1. Ensure backward compatibility is maintained
2. Add appropriate error handling and logging
3. Update tests to cover new functionality
4. Verify performance improvements with benchmarks

## Troubleshooting

### Issue: Parallel rollout slower than sequential

**Solutions:**
- Reduce number of workers (`parallel_n_jobs`)
- Check if environments are too simple/fast
- Verify system has multiple CPU cores available

### Issue: Memory errors with parallel rollout

**Solutions:**
- Reduce `parallel_n_jobs`
- Reduce environment batch sizes
- Check GPU memory usage

### Issue: Inconsistent results

**Solutions:**
- Verify random seed handling
- Check environment determinism
- Compare with sequential rollout results

For additional help, please check the error logs or create an issue with detailed information about your setup and error messages.