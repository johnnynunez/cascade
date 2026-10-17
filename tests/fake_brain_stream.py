"""What the openai client returns for `chat.completions.create(..., stream=True)`,
for the brain detector's fake clients.

The brain detector streams its BACKGROUND requests (watcher inventories and
vocabulary refreshes) so a foreground localization can cancel them: the reply
arrives as chunks whose `choices[0].delta.content` carry the text and whose
last choice carries `finish_reason`; `close()` ends the HTTP response, which
makes vLLM abort the request. Foreground refers stay non-streaming.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace


def chunk(content: str | None, finish_reason: str | None = None) -> SimpleNamespace:
    """One ChatCompletionChunk: a text delta, or the closing finish reason."""
    choice = SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class FakeStream:
    """An openai Stream over `text`: while `hold` (an Event) is unset it keeps
    "decoding" (an empty delta every `tick_s`, like tokens of a long reply),
    then yields the text in a few pieces and the finish reason. Iteration ends
    once the stream is closed; `on_close` runs at the first close()."""

    def __init__(self, text: str, finish_reason: str = "stop", *, hold: threading.Event | None = None,
                 tick_s: float = 0.01, on_close=None):
        self.text, self.finish_reason = text, finish_reason
        self.hold, self.tick_s, self._on_close = hold, tick_s, on_close
        self.closed = False
        self.exhausted = False

    def __iter__(self):
        while self.hold is not None and not self.hold.is_set():
            if self.closed:
                return
            yield chunk("")
            time.sleep(self.tick_s)
        step = max(1, len(self.text) // 3)
        for i in range(0, len(self.text), step):
            if self.closed:
                return
            yield chunk(self.text[i:i + step])
        if not self.closed:
            self.exhausted = True
            yield chunk(None, self.finish_reason)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            if self._on_close is not None:
                self._on_close(self)


def completion(text: str, finish_reason: str = "stop") -> SimpleNamespace:
    """A non-streaming ChatCompletion with one choice."""
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def reply(kwargs: dict, text: str, finish_reason: str = "stop"):
    """The client's answer to one create(**kwargs): a FakeStream when the
    request streams, else a ChatCompletion."""
    if kwargs.get("stream"):
        return FakeStream(text, finish_reason)
    return completion(text, finish_reason)
