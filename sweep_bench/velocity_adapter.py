from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sweep_bench.data.frame_builder import SENSOR_COMPOSITE_LAYOUT, build_sensor_composite
from sweep_bench.data.schemas import get_schema

if TYPE_CHECKING:
    from pathlib import Path

    from sweep_bench.policies.base import Obs

SPEED_CENTER = 0.25
SPEED_SCALE = 0.15
FLOW_CNN_POOL = 2
FLOW_SCALE_PX = 4.0


class ImageFlow:
    def __init__(self, device: str = "cpu", flow_updates: int = 12, upsample: int = 2, out_side: int = 112):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.device = device
        self.flow_updates = flow_updates
        self.upsample = upsample
        self.out_side = out_side
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).eval().to(device)

    @torch.no_grad()
    def __call__(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        pair = []
        for frames in (first, second):
            t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().to(self.device)
            pair.append(
                F.interpolate(t / 127.5 - 1.0, scale_factor=self.upsample, mode="bilinear", align_corners=False)
            )
        flow = self.model(pair[0], pair[1], num_flow_updates=self.flow_updates)[-1] / self.upsample
        return F.adaptive_avg_pool2d(flow, self.out_side).cpu().numpy()


class PooledCNN(nn.Module):
    def __init__(self, in_channels: int, width: int = 16, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, logit = self.body(x).unbind(dim=1)
        weight = torch.softmax(logit.flatten(1), dim=1)
        return (weight * value.flatten(1)).sum(dim=1)


def flow_cnn_inputs(flow: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(flow, dtype=np.float32))
    t = F.avg_pool2d(t, FLOW_CNN_POOL).abs() / FLOW_SCALE_PX
    s, _, h, w = t.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    coords = torch.stack([xs, ys]).expand(s, 2, h, w)
    return torch.cat([t, coords], dim=1)


def exterior_composite(obs: Obs) -> np.ndarray:
    sensor_data = cast("dict[str, dict[str, torch.Tensor]]", obs["sensor_data"])
    raw = {
        uid: sensor_data[uid]["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        for row in SENSOR_COMPOSITE_LAYOUT
        for uid in row
    }
    schema = get_schema("droid_panda_v3")
    return build_sensor_composite(raw, schema.image_height, schema.image_width)


class OnlineBeltSpeed:
    def __init__(self, head: Path, device: str, gap: int = 2, pairs: int = 5, prior: float = SPEED_CENTER):
        self.device = device
        self.flow = ImageFlow(device=device)
        self.head = PooledCNN(4, stride=2)
        self.head.load_state_dict(torch.load(head, map_location="cpu", weights_only=True))
        self.head = self.head.eval().to(device)
        self.gap = gap
        self.pairs = pairs
        self.prior = prior
        self._frames: deque[np.ndarray] = deque(maxlen=gap + 1)
        self.pair_estimates: list[float] = []
        self.sent: list[float] = []

    def reset(self) -> None:
        self._frames.clear()
        self.pair_estimates = []
        self.sent = []

    @torch.no_grad()
    def __call__(self, composite: np.ndarray) -> float:
        self._frames.append(composite)
        if len(self._frames) > self.gap:
            flow = self.flow(self._frames[0][None], self._frames[-1][None])
            normalized = self.head(flow_cnn_inputs(flow).to(self.device)).item()
            self.pair_estimates.append(normalized * SPEED_SCALE + SPEED_CENTER)
        recent = self.pair_estimates[-self.pairs :]
        speed = float(np.mean(recent)) if recent else self.prior
        self.sent.append(speed)
        return speed

    def summary(self, true_speed: float) -> dict[str, float | int | list[float] | None]:
        sent = np.asarray(self.sent[self.gap :])
        pairs = np.asarray(self.pair_estimates)
        return {
            "adapter_first_estimate_step": self.gap if len(pairs) else None,
            "adapter_mean": float(sent.mean()) if len(sent) else None,
            "adapter_mae": float(np.abs(sent - true_speed).mean()) if len(sent) else None,
            "adapter_pair_mae": float(np.abs(pairs - true_speed).mean()) if len(pairs) else None,
            "adapter_sent": [round(v, 4) for v in self.sent],
            "adapter_pair_estimates": [round(v, 4) for v in self.pair_estimates],
        }
