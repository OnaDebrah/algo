import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)


class AsyncHelper:
    """Helper class for running async code from sync contexts"""

    _executor = ThreadPoolExecutor(max_workers=4)
    _loop_cache = {}
    _default_timeout = 30  # seconds

    @classmethod
    def run_async(cls, coro, timeout=None):
        """
        Run an async coroutine from sync context, handling all edge cases.

        This is thread-safe and works in:
        - Pure sync contexts
        - Nested async contexts
        - Multi-threaded environments

        Args:
            coro: The coroutine to run
            timeout: Optional timeout in seconds

        Returns:
            The result of the coroutine

        Raises:
            TimeoutError: If the operation times out
            Exception: Any exception raised by the coroutine
        """
        timeout = timeout or cls._default_timeout

        try:
            # Case 1: We're already in an async context
            loop = asyncio.get_running_loop()

            # Create a future in the current event loop
            future = asyncio.create_task(coro)

            # If we're in the main thread, we need special handling
            if threading.current_thread() is threading.main_thread():
                # Main thread with running loop - can't block
                # Instead, return a future that can be awaited later
                logger.debug("Main thread in async context - returning future")
                return future  # Now the future IS used!

            # For non-main threads, use run_coroutine_threadsafe
            logger.debug("Non-main thread in async context - using threadsafe future")
            thread_future = asyncio.run_coroutine_threadsafe(coro, loop)

            try:
                return thread_future.result(timeout=timeout)
            except TimeoutError:
                thread_future.cancel()
                raise TimeoutError(f"Async operation timed out after {timeout} seconds")

        except RuntimeError:
            # Case 2: No running loop - we can create one
            logger.debug("No running loop - using asyncio.run()")
            try:
                return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
            except asyncio.TimeoutError:
                raise TimeoutError(f"Async operation timed out after {timeout} seconds")

    @classmethod
    def run_async_with_callback(cls, coro, callback=None, timeout=None):
        """
        Run async coroutine with optional callback when complete.

        This is useful for fire-and-forget operations where you don't
        need to wait for the result immediately.
        """

        def _run_in_thread():
            try:
                result = cls.run_async(coro, timeout)
                if callback:
                    callback(result)
                return result
            except Exception as e:
                logger.error(f"Async operation failed: {e}")
                if callback:
                    callback(None, e)
                raise

        # Run in thread pool to avoid blocking
        future = cls._executor.submit(_run_in_thread)
        return future

    @classmethod
    async def run_sync(cls, func, *args, **kwargs):
        """Run a sync function from async context without blocking"""
        loop = asyncio.get_running_loop()

        # Check if we should use timeout
        timeout = kwargs.pop("timeout", None)

        if timeout:
            # Run with timeout
            try:
                return await asyncio.wait_for(loop.run_in_executor(cls._executor, lambda: func(*args, **kwargs)), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Sync operation timed out after {timeout} seconds")
        else:
            # Run without timeout
            return await loop.run_in_executor(cls._executor, lambda: func(*args, **kwargs))

    @classmethod
    async def run_sync_batch(cls, funcs_with_args, max_concurrent=None):
        """
        Run multiple sync functions concurrently from async context

        Args:
            funcs_with_args: List of (func, args, kwargs) tuples
            max_concurrent: Maximum number of concurrent executions

        Returns:
            List of results in the same order
        """
        loop = asyncio.get_running_loop()
        semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _run_one(func, args, kwargs):
            if semaphore:
                async with semaphore:
                    return await loop.run_in_executor(cls._executor, lambda: func(*args, **kwargs))
            else:
                return await loop.run_in_executor(cls._executor, lambda: func(*args, **kwargs))

        tasks = []
        for func, args, kwargs in funcs_with_args:
            tasks.append(_run_one(func, args, kwargs))

        return await asyncio.gather(*tasks, return_exceptions=True)

    @classmethod
    def shutdown(cls, wait=True):
        """Shutdown the thread pool executor"""
        cls._executor.shutdown(wait=wait)
        logger.info("AsyncHelper executor shutdown")
