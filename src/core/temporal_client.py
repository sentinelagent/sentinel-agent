import asyncio
from temporalio.client import Client
from src.core.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Temporal connection retry configuration
TEMPORAL_MAX_RETRIES = 30  # Keep trying for a long time
TEMPORAL_BASE_DELAY = 2.0  # Start with 2 seconds
TEMPORAL_MAX_DELAY = 60.0  # Cap at 60 seconds between retries


async def connect_to_temporal_with_retry(
    target_host: str | None = None,
    namespace: str = "default",
    max_retries: int = TEMPORAL_MAX_RETRIES,
    base_delay: float = TEMPORAL_BASE_DELAY,
    max_delay: float = TEMPORAL_MAX_DELAY,
) -> Client:
    """
    Connect to Temporal server with retry and exponential backoff.

    This function will keep retrying the connection if Temporal is not available,
    which is useful during startup when Temporal may still be initializing.

    Args:
        target_host: Temporal server URL (defaults to settings.TEMPORAL_SERVER_URL)
        namespace: Temporal namespace (default: "default")
        max_retries: Maximum number of retry attempts (default: 30)
        base_delay: Initial delay in seconds before first retry (default: 2.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)

    Returns:
        Connected Temporal Client

    Raises:
        Exception: If all retries are exhausted
    """
    if target_host is None:
        target_host = getattr(settings, 'TEMPORAL_SERVER_URL', 'localhost:7233')

    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            client = await Client.connect(
                target_host=target_host,
                namespace=namespace,
            )
            logger.info(f"Successfully connected to Temporal server at {target_host}")
            return client

        except Exception as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    f"Failed to connect to Temporal after {max_retries + 1} attempts: {e}"
                )
                raise

            # Calculate delay with exponential backoff (capped at max_delay)
            delay = min(base_delay * (2 ** attempt), max_delay)

            logger.warning(
                f"Temporal connection attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            await asyncio.sleep(delay)

    # This should never be reached, but satisfies type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in connect_to_temporal_with_retry")


class TemporalClient:
    """
    Singleton Temporal client with automatic connection retry.

    This class provides a shared Temporal client connection that can be used
    across the application. It automatically retries connection if Temporal
    is not available.
    """

    def __init__(self):
        self.client: Client | None = None

    async def connect(self):
        """Connect to Temporal server with retry logic."""
        temporal_host = getattr(settings, 'TEMPORAL_SERVER_URL', 'localhost:7233')
        self.client = await connect_to_temporal_with_retry(
            target_host=temporal_host,
            namespace="default",
        )

    async def disconnect(self):
        """Disconnect from Temporal server."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Successfully disconnected from Temporal server.")

    async def get_client(self) -> Client:
        """
        Get the Temporal client, connecting if necessary.

        Returns:
            Connected Temporal Client
        """
        if not self.client:
            await self.connect()
        return self.client


temporal_client = TemporalClient()
