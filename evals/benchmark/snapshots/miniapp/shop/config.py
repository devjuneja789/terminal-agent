"""Application configuration."""

import os


DEFAULT_TIMEOUT_SECONDS = 10
MAX_PAGE_SIZE = 100
REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT_SECONDS"


def request_timeout() -> int:
    """Read the request timeout, falling back to the documented default."""
    return DEFAULT_TIMEOUT_SECONDS


def page_size(value: int) -> int:
    """Clamp a page size to the configured range."""
    return min(value, MAX_PAGE_SIZE)
