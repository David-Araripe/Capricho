"""Module for rate limiting function calls through a decorator."""

import threading
import time
from functools import wraps

from ..logger import logger


def rate_limit(max_per_second=5):
    """Decorator to rate limit function calls.

    Args:
        max_per_second: max amount of API calls to be started within a second. Each call
            reserves the next free start slot and sleeps until it comes up. Defaults to 5.

    Returns:
        Decorator: paces the decorated function. Reusing one decorator across several
        functions makes them share a single budget.
    """
    min_interval = 1.0 / max_per_second
    lock = threading.Lock()

    next_allowed_start_time = 0.0

    logger.debug(
        f"Initializing rate limiter: max {max_per_second} calls per second, min interval {min_interval:.4f}s"
    )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal next_allowed_start_time
            # Only the slot reservation is under the lock, so concurrent calls can overlap.
            with lock:
                current_attempt_time = time.monotonic()
                allowed_start_time = max(current_attempt_time, next_allowed_start_time)
                next_allowed_start_time = allowed_start_time + min_interval
                time_to_wait = allowed_start_time - current_attempt_time

            if time_to_wait > 0:
                logger.trace(f"Rate limit exceeded. Waiting for {time_to_wait:.4f}s")
                time.sleep(time_to_wait)

            try:
                logger.trace(  # Log the exact time it's starting execution for test validation
                    f"Function {func.__name__} starting execution at {allowed_start_time:.4f}"
                )
                ret = func(*args, **kwargs)
                logger.trace(f"Finished executing {func.__name__}")
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise
            return ret

        return wrapper

    return decorator
