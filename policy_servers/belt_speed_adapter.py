import functools
import logging
import queue
import threading
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

METADATA_KEY = "belt_speed_adapter"
EPISODE_KEY = "belt_speed_episode"
SPEED_KEY = "belt_speed"
FRAME_PUSH = b"__frame_push__"
SPEED_MIN, SPEED_MAX = 0.0, 0.99
FORMAT = "belt_speed_adapter/1"
FLOW_CNN_POOL = 2
FLOW_SCALE_PX = 4.0
EPISODES_KEPT = 4


class PooledCNN(nn.Module):
    def __init__(self, in_channels: int, width: int = 16, stride: int = 2):
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


class DeepPooledCNN(PooledCNN):
    def __init__(self, in_channels: int, width: int = 32):
        nn.Module.__init__(self)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 2, 1),
        )


HEADS = {
    "flow_cnn": lambda: PooledCNN(4),
    "flow_cnn_w64": lambda: PooledCNN(4, width=64),
    "flow_cnn_deep": lambda: DeepPooledCNN(4),
}


def save_head(
    path, head: str, state_dicts: list[dict], center: float, scale: float, flow: dict, gap: int, provenance: dict
) -> None:
    if head not in HEADS:
        raise ValueError(f"unknown head {head!r}; one of {sorted(HEADS)}")
    torch.save(
        {
            "format": FORMAT,
            "head": head,
            "state_dicts": state_dicts,
            "center": center,
            "scale": scale,
            "flow": flow,
            "gap": gap,
            "provenance": provenance,
        },
        path,
    )


def load_head(path) -> dict:
    spec = torch.load(path, map_location="cpu", weights_only=True)
    if spec.get("format") != FORMAT:
        raise ValueError(f"{path}: format {spec.get('format')!r}, not {FORMAT!r}")
    return spec


def flow_cnn_inputs(flow: torch.Tensor) -> torch.Tensor:
    t = F.avg_pool2d(flow, FLOW_CNN_POOL).abs() / FLOW_SCALE_PX
    s, _, h, w = t.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=t.device), torch.linspace(-1, 1, w, device=t.device), indexing="ij"
    )
    return torch.cat([t, torch.stack([xs, ys]).expand(s, 2, h, w)], dim=1)


class FlowSpeed:
    def __init__(self, spec: dict, device: str, cuda_graph: bool = False):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.updates = int(spec["flow"]["updates"])
        self.upsample = int(spec["flow"]["upsample"])
        self.out_side = int(spec["flow"]["out_side"])
        self.center, self.scale = float(spec["center"]), float(spec["scale"])
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT).eval().to(self.device)
        self.heads = []
        for state in spec["state_dicts"]:
            head = HEADS[spec["head"]]()
            head.load_state_dict(state)
            self.heads.append(head.eval().to(self.device))
        self.cuda_graph = cuda_graph
        self._graph = None

    def _normalized(self, pair: torch.Tensor) -> torch.Tensor:
        x = pair.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        x = F.interpolate(x, scale_factor=self.upsample, mode="bilinear", align_corners=False)
        flow = self.raft(x[:1], x[1:], num_flow_updates=self.updates)[-1] / self.upsample
        inputs = flow_cnn_inputs(F.adaptive_avg_pool2d(flow, self.out_side))
        return torch.stack([head(inputs) for head in self.heads]).mean(dim=0)

    @torch.no_grad()
    def __call__(self, first: np.ndarray, second: np.ndarray) -> float:
        pair = np.stack([first, second])
        if not self.cuda_graph:
            return float(self._normalized(torch.from_numpy(pair).to(self.device)).item()) * self.scale + self.center
        if self._graph is None:
            self._capture(pair.shape)
        self._input.copy_(torch.from_numpy(pair), non_blocking=False)
        self._graph.replay()
        return float(self._output.item()) * self.scale + self.center

    def _capture(self, shape: tuple[int, ...]) -> None:
        import torchvision.models.optical_flow.raft as raft_module

        self._input = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        make_coords_grid, linspace = raft_module.make_coords_grid, torch.linspace
        raft_module.make_coords_grid = functools.partial(make_coords_grid, device=self.device)
        torch.linspace = functools.partial(linspace, device=self.device)
        try:
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._normalized(self._input)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._output = self._normalized(self._input)
        finally:
            raft_module.make_coords_grid, torch.linspace = make_coords_grid, linspace


class Session:
    def __init__(self, adapter: "BeltSpeedAdapter"):
        self._adapter = adapter
        self._lock = threading.Lock()
        self._frames: dict[int, OrderedDict[int, np.ndarray]] = {}
        self._estimates: OrderedDict[int, list[float]] = OrderedDict()

    def observe(self, message: Mapping) -> None:
        if FRAME_PUSH not in message:
            return
        image = message[b"images"].get(self._adapter.image_key)
        if image is not None:
            self._adapter.submit(self, int(message[b"episode"]), int(message[b"step"]), np.array(image))

    def add(self, episode: int, step: int, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        gap = self._adapter.gap
        frames = self._frames.setdefault(episode, OrderedDict())
        frames[step] = image
        while len(frames) > 2 * gap + 2:
            frames.popitem(last=False)
        while len(self._frames) > EPISODES_KEPT:
            self._frames.pop(min(self._frames))
        earlier = frames.get(step - gap)
        return None if earlier is None else (earlier, image)

    def record(self, episode: int, speed: float) -> None:
        with self._lock:
            self._estimates.setdefault(episode, []).append(speed)
            while len(self._estimates) > EPISODES_KEPT:
                self._estimates.popitem(last=False)

    def fill(self, obs: Mapping) -> tuple[dict, dict]:
        if self._adapter.error is not None:
            raise RuntimeError("the belt-speed adapter's worker failed") from self._adapter.error
        if SPEED_KEY in obs:
            raise ValueError(
                f"this server estimates {SPEED_KEY!r} itself (--belt-speed-adapter); send {EPISODE_KEY!r} instead"
            )
        if EPISODE_KEY not in obs:
            raise ValueError(
                f"a --belt-speed-adapter server needs {EPISODE_KEY!r} (the rollout id of the frame pushes)"
            )
        out = dict(obs)
        episode = int(out.pop(EPISODE_KEY))
        with self._lock:
            recent = self._estimates.get(episode, [])[-self._adapter.pairs :]
            total = len(self._estimates.get(episode, []))
        raw = float(np.mean(recent)) if recent else self._adapter.prior
        speed = min(max(raw, SPEED_MIN), SPEED_MAX)
        out[SPEED_KEY] = speed
        return out, {"speed": speed, "raw": raw, "pairs": len(recent), "estimates": total}


class BeltSpeedAdapter:
    reply_key = METADATA_KEY

    def __init__(
        self,
        path,
        image_key: str,
        device: str = "cuda",
        pairs: int = 5,
        prior: float | None = None,
        cuda_graph: bool = False,
    ):
        spec = load_head(path)
        self.image_key = image_key
        self.gap = int(spec["gap"])
        self.pairs = pairs
        self.prior = float(spec["center"]) if prior is None else prior
        self.flow = FlowSpeed(spec, device, cuda_graph)
        self.error: BaseException | None = None
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        self.flow(blank, blank)
        self.info = {
            "head": spec["head"],
            "heads": len(spec["state_dicts"]),
            "flow": spec["flow"],
            "gap": self.gap,
            "pairs": pairs,
            "prior": self.prior,
            "cuda_graph": cuda_graph,
        }
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, name="belt-speed-adapter", daemon=True).start()

    def metadata(self) -> dict:
        return {METADATA_KEY: self.info}

    def session(self) -> Session:
        return Session(self)

    def submit(self, session: Session, episode: int, step: int, image: np.ndarray) -> None:
        self._queue.put((session, episode, step, image))

    def drain(self) -> None:
        self._queue.join()

    def _worker(self) -> None:
        while True:
            session, episode, step, image = self._queue.get()
            try:
                pair = session.add(episode, step, image)
                if pair is not None:
                    session.record(episode, self.flow(*pair))
            except BaseException as e:  # noqa: BLE001 - every later request on any connection refuses with it
                logging.exception("belt-speed adapter worker failed")
                self.error = e
                return
            finally:
                self._queue.task_done()
