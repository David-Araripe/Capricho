"""Module for rate limiting function calls through a decorator."""

import threading
import time
from functools import wraps

from ..logger import logger


def rate_limit(max_per_second=5):
    """Decorator to rate limit function calls.

    Args:
        max_per_second: max amount of API calls to be done within a second. Setting this
            parameter will cause the coming function call to sleep in case the previous
            call was done less than 1/max_per_second seconds ago. Defaults to 5.

    Returns:
        Decorator: Function decorator that will rate limit the decorated function
    """
    min_interval = 1.0 / max_per_second
    lock = threading.Lock()

    last_allowed_start_time = 0

    logger.debug(
        f"Initializing rate limiter: max {max_per_second} calls per second, min interval {min_interval:.4f}s"
    )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_allowed_start_time
            with lock:
                current_attempt_time = time.time()
                elapsed_since_last_allowed_start = current_attempt_time - last_allowed_start_time
                time_to_wait = min_interval - elapsed_since_last_allowed_start

                logger.trace(
                    f"Function {func.__name__} called. Time since last allowed start: {elapsed_since_last_allowed_start:.4f}s"
                )

                if time_to_wait > 0:
                    logger.trace(f"Rate limit exceeded. Waiting for {time_to_wait:.4f}s")
                    time.sleep(time_to_wait)
                    # The *actual* time this function is allowed to start is
                    # current_attempt_time + time_to_wait
                    last_allowed_start_time = current_attempt_time + time_to_wait
                else:
                    logger.trace("Executing immediately")
                    # No wait needed, so the current attempt time is the allowed start time
                    last_allowed_start_time = current_attempt_time

                try:
                    logger.trace(  # Log the exact time it's starting execution for test validation
                        f"Function {func.__name__} starting execution at {last_allowed_start_time:.4f}"
                    )
                    ret = func(*args, **kwargs)
                    logger.trace(f"Finished executing {func.__name__}")
                except Exception as e:
                    logger.exception(f"Exception in {func.__name__}: {e!s}")
                    raise
            return ret

        return wrapper

    return decorator
