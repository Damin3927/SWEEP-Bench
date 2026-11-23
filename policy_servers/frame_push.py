from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

PUSH = b"__frame_push__"
REFS = b"__frame_refs__"
METADATA_KEY = "frame_push"
CAPACITY = 64
OFF = (
    "this server was started without --frame-push and takes no frame pushes or frame refs; a client that does not "
    f"find {METADATA_KEY!r} in the connect metadata must send full requests"
)


def metadata(on: bool) -> dict:
    return {METADATA_KEY: {"capacity": CAPACITY}} if on else {}


class FrameStore:
    def __init__(self, on: bool, capacity: int = CAPACITY) -> None:
        self._on = on
        self._capacity = capacity
        self._frames: OrderedDict[tuple[int, int], Mapping] = OrderedDict()

    def take(self, message: Mapping) -> Mapping | None:
        if PUSH in message:
            if not self._on:
                raise ValueError(OFF)
            frame = (int(message[b"episode"]), int(message[b"step"]))
            self._frames.pop(frame, None)
            self._frames[frame] = message[b"images"]
            while len(self._frames) > self._capacity:
                self._frames.popitem(last=False)
            return None
        keys = [k for k, v in message.items() if isinstance(v, Mapping) and REFS in v]
        if not keys:
            return message
        if not self._on:
            raise ValueError(OFF)
        out = dict(message)
        for key in keys:
            out[key] = np.stack([self._frame(key, item) for item in message[key][b"frames"]])
        return out

    def _frame(self, key: str, item) -> np.ndarray:
        if isinstance(item, np.ndarray):
            return item
        episode, step = (int(x) for x in item)
        images = self._frames.get((episode, step))
        if images is None or key not in images:
            raise LookupError(f"{key}: no pushed frame for episode {episode}, step {step} on this connection")
        return images[key]
