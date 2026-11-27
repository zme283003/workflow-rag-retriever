"""Progress reporting (fire-and-forget style).

The retrieval workflow emits human-readable stage updates so a UI can show
what it is doing. Implementations must never break retrieval on failure.
"""

from __future__ import annotations

from typing import Callable, Optional


class NullProgressSink:
    def report(self, key: str, message: str) -> None:
        return None


class CallbackProgressSink:
    """Delegates to a user-supplied callback, swallowing all errors."""

    def __init__(self, callback: Callable[[str, str], None]):
        self.callback = callback

    def report(self, key: str, message: str) -> None:
        try:
            self.callback(key, message)
        except Exception:
            pass
