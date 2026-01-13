"""Caching utilities for performance optimization."""

import hashlib
import json
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from trading_lab.config.settings import get_settings

T = TypeVar("T")


def file_cache(
    cache_dir: Optional[Path] = None,
    max_age_days: Optional[int] = None,
    suffix: str = ".cache",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to cache function results to disk.

    Args:
        cache_dir: Directory to store cache files. If None, uses default cache directory.
        max_age_days: Maximum age of cache file in days. If None, cache never expires.
        suffix: File suffix for cache files.

    Returns:
        Decorated function with caching enabled.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Determine cache directory
            if cache_dir is None:
                settings = get_settings()
                cache_path = settings.get_artifacts_dir() / "cache"
            else:
                cache_path = cache_dir
            
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Generate cache key from function name and arguments
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            cache_file = cache_path / f"{cache_key}{suffix}"
            
            # Check if cache exists and is valid
            if cache_file.exists():
                if max_age_days is None:
                    # Cache is valid
                    try:
                        with open(cache_file, "rb") as f:
                            return pickle.load(f)
                    except Exception as e:
                        # Cache file is corrupted, remove it
                        cache_file.unlink()
                else:
                    # Check cache age
                    from datetime import datetime, timedelta
                    file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_age < timedelta(days=max_age_days):
                        try:
                            with open(cache_file, "rb") as f:
                                return pickle.load(f)
                        except Exception:
                            cache_file.unlink()
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except Exception as e:
                # If caching fails, continue without cache
                pass
            
            return result
        
        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from function name and arguments."""
    # Convert arguments to a hashable format
    key_data = {
        "func": func_name,
        "args": str(args),
        "kwargs": json.dumps(kwargs, sort_keys=True, default=str),
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()
