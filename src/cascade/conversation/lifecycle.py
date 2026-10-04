"""Attempt each conversation owner without upgrading a partial close."""
from __future__ import annotations

import asyncio

from ..lifecycle import teardown_step


async def close_stage(name, operation):
    try:
        result = await operation()
    except (Exception, asyncio.CancelledError) as error:
        # Provider errors can contain URLs or credentials. Preserve the type,
        # not external exception text, in the local service receipt.
        return {"stage": name, "ok": False, "complete": False,
                "errors": [{"type": type(error).__name__}]}
    return teardown_step(name, lambda: result)
