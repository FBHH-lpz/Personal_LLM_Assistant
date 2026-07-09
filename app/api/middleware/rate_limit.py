"""Simple token-bucket rate limiter middleware."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class TokenBucketRateLimiter(BaseHTTPMiddleware):
    """Per-IP token bucket rate limiter.

    Args:
        rate: Number of tokens (requests) per window.
        window: Time window in seconds.
    """

    def __init__(self, app, rate: int = 60, window: int = 60):
        super().__init__(app)
        self.rate = rate
        self.window = window
        self._buckets: dict[str, tuple[int, float]] = defaultdict(
            lambda: (rate, time.time())
        )

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        tokens, last_refill = self._buckets[client_ip]
        now = time.time()

        # Refill tokens
        elapsed = now - last_refill
        new_tokens = int(elapsed * (self.rate / self.window))
        tokens = min(self.rate, tokens + new_tokens)
        last_refill = now if new_tokens > 0 else last_refill

        if tokens <= 0:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait."},
            )

        tokens -= 1
        self._buckets[client_ip] = (tokens, last_refill)

        return await call_next(request)
