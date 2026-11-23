from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping

SPEED_KEY = "belt_speed"
SPEED_SLOT = 6
SPEED_CENTER = 0.25
SPEED_SCALE = 0.15
STAMP = "bench_speed_token.json"
STAMP_VALUE = {"slot": SPEED_SLOT, "center": SPEED_CENTER, "scale": SPEED_SCALE}


def encode_speed(speed):
    return (speed - SPEED_CENTER) / SPEED_SCALE


def inject_speed(state, speed):
    out = state.clone()
    out[..., SPEED_SLOT] = encode_speed(speed.to(device=out.device, dtype=out.dtype).reshape(out.shape[:-1]))
    return out


def zero_speed_column(state_proj) -> None:
    import torch

    with torch.no_grad():
        state_proj.weight[:, SPEED_SLOT] = 0.0


def patch_prepare_state(policy_cls) -> None:
    import torch

    original = policy_cls.prepare_state

    def prepare_state(self, batch):
        if SPEED_KEY not in batch:
            raise KeyError(f"speed-conditioned DynamicVLA got a batch without {SPEED_KEY!r}")
        raw_dim = batch["observation.state"].shape[-1]
        if raw_dim > SPEED_SLOT:
            raise ValueError(f"state is {raw_dim}-dim, so slot {SPEED_SLOT} is not padding")
        state = original(self, batch)
        return inject_speed(state, torch.as_tensor(batch[SPEED_KEY]))

    policy_cls.prepare_state = prepare_state


def check_request(speed_conditioned: bool, request: Mapping[str, object]) -> None:
    if speed_conditioned != (SPEED_KEY in request):
        raise ValueError(
            f"checkpoint speed_conditioned={speed_conditioned} but the request "
            f"{'carries' if SPEED_KEY in request else 'lacks'} {SPEED_KEY!r}; "
            "serve and evaluate with matching --speed-token"
        )


def find_stamp(ckpt_dir: pathlib.Path) -> pathlib.Path | None:
    return next((p / STAMP for p in (ckpt_dir, *ckpt_dir.parents) if (p / STAMP).is_file()), None)


def read_stamp(path: pathlib.Path) -> None:
    recorded = json.loads(path.read_text()).get("speed_token")
    if recorded != STAMP_VALUE:
        raise RuntimeError(f"{path} records speed_token={recorded!r}; this code encodes {STAMP_VALUE!r}")
