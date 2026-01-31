"""
Retry utilities with exponential backoff for handling transient failures.

This module provides retry functionality specifically designed for network-related
errors that are often transient and can succeed on retry.
"""

import asyncio
import random
from typing import Callable, TypeVar, Any, Tuple, Type
from functools import wraps

from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Default network-related exceptions to retry
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    ConnectionResetError,
    TimeoutError,
    OSError,
)


async def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Callable[[Exception, int, float], None] | None = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with exponential backoff retry logic.

    Args:
        func: The async function to execute
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delay to avoid thundering herd (default: True)
        retryable_exceptions: Tuple of exception types that should trigger a retry
        on_retry: Optional callback called before each retry with (exception, attempt, delay)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            if on_retry:
                on_retry(e, attempt + 1, delay)

            await asyncio.sleep(delay)

    # This should never be reached, but satisfies type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry_with_backoff")


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
):
    """
    Decorator to add retry with exponential backoff to an async function.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delay (default: True)
        retryable_exceptions: Tuple of exception types that should trigger a retry

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
                **kwargs,
            )
        return wrapper
    return decorator
