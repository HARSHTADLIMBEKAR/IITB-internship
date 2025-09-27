"""
Device and reproducibility utilities.
"""

import torch
import random
import numpy as np
import os
from typing import Optional


def get_device(use_cuda: bool = True, device_id: int = 0) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        use_cuda: Whether to use CUDA
        device_id: CUDA device ID
        
    Returns:
        PyTorch device
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        info['cuda_devices'] = []
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i)
            }
            info['cuda_devices'].append(device_info)
    
    return info


def print_device_info():
    """Print information about available devices."""
    info = get_device_info()
    
    print("Device Information:")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Device Count: {info['cuda_device_count']}")
        print(f"Current Device: {info['current_device']}")
        
        for device in info['cuda_devices']:
            print(f"\nDevice {device['id']}: {device['name']}")
            print(f"  Total Memory: {device['memory_total'] / 1024**3:.2f} GB")
            print(f"  Allocated Memory: {device['memory_allocated'] / 1024**3:.2f} GB")
            print(f"  Cached Memory: {device['memory_cached'] / 1024**3:.2f} GB")
    else:
        print("CUDA not available, using CPU")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")


def get_memory_usage() -> dict:
    """
    Get current memory usage.
    
    Returns:
        Dictionary containing memory usage information
    """
    memory_info = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info[f'cuda_{i}'] = {
                'allocated': torch.cuda.memory_allocated(i),
                'cached': torch.cuda.memory_reserved(i),
                'max_allocated': torch.cuda.max_memory_allocated(i),
                'max_cached': torch.cuda.max_memory_reserved(i)
            }
    
    return memory_info


def print_memory_usage():
    """Print current memory usage."""
    memory_info = get_memory_usage()
    
    if memory_info:
        print("Memory Usage:")
        for device, info in memory_info.items():
            print(f"{device}:")
            print(f"  Allocated: {info['allocated'] / 1024**3:.2f} GB")
            print(f"  Cached: {info['cached'] / 1024**3:.2f} GB")
            print(f"  Max Allocated: {info['max_allocated'] / 1024**3:.2f} GB")
            print(f"  Max Cached: {info['max_cached'] / 1024**3:.2f} GB")
    else:
        print("No CUDA devices available")
