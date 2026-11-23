from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_edge_recenter.config import RandomizationConfig
from sweep_bench.tasks.conveyor_edge_recenter.scripted import ScriptedEdgeRecenterMixin
from sweep_bench.tasks.conveyor_edge_recenter.variants import (
    EDGE_RECENTER_OBJECT_IDS,
    PARCEL_SPEC,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorEdgeRecenter-v0")
@register_env("ConveyorEdgeRecenter-v0", max_episode_steps=300)
class ConveyorEdgeRecenterEnv(ScriptedEdgeRecenterMixin, ConveyorWorkspaceEnv):
    """Re-seat a box that is riding the belt HANGING OVER a side rail."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(EDGE_RECENTER_OBJECT_IDS, extra=(PARCEL_SPEC,))

    _overhang_frac_default: float = 0.35
    _overhang_side_default: float = -1.0
    _spawn_end_margin: float = 0.04

    _finger_below_tcp: float = 0.009

    _part_patch_radius: float = 0.02

    _band_half: float = 0.08
    _center_tol: float = 0.030
    _deck_rest_tol: float = 0.006
    _tip_max_tilt: float = math.radians(15.0)
    _comove_tol: float = 0.06
    _lateral_speed_tol: float = 0.05
    _released_aperture: float = 0.070
    _released_distance: float = 0.10
    _runway_min: float = 0.25
    _off_deck_drop: float = 0.02
    _recenter_sustain_steps: int = 12

    default_prompt: str = "move the overhanging box back onto the middle of the conveyor belt"

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"move the overhanging {name} back onto the middle of the conveyor belt" for name in self._part_display]

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        lo, hi = self.randomization.overhang_frac_range
        fracs = (self._overhang_frac_default, lo, hi)
        if not all(0.0 < f < 0.5 for f in fracs):
            raise ValueError(
                f"overhang fractions must lie in (0, 0.5) so the box's COM stays over the deck, got {fracs}"
            )
        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        specs = [spec_by_id[o] for o in self.objects]

        widest = max(spec.bbox[0] for spec in specs)
        if 2.0 * widest >= self._released_aperture:
            raise ValueError(
                f"box cross-belt width {2.0 * widest} m is not below the released jaw opening "
                f"{self._released_aperture} m: a gripping hand would read as open"
            )
        shortest = min(spec.bbox[2] for spec in specs)
        if shortest <= self._finger_below_tcp:
            raise ValueError(
                f"box half-height {shortest} m is not above the fingertip drop below the TCP "
                f"({self._finger_below_tcp} m): the grasp would drive the fingertips into the deck"
            )
        for spec in specs:
            for frac in (self._overhang_frac_default, lo, hi):
                spawn_dx = self._band_half - spec.bbox[0] * (1.0 - 2.0 * frac)
                if spawn_dx <= self._center_tol or spawn_dx + spec.bbox[0] <= self._band_half:
                    raise ValueError(
                        f"{spec.id} spawns at |x - ox| = {spawn_dx:.4f} m with overhang_frac={frac}: "
                        f"that already satisfies the success band (centre tol {self._center_tol}, "
                        f"band half {self._band_half})"
                    )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        device = self.device

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        part_objs: list[Actor] = []
        patch_radii: list[float] = []
        halves: list[tuple[float, float, float]] = []
        for i, name in enumerate(self._part_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, oz + self._conveyor_deck_height + 0.2])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            part_objs.append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._part_patch_radius)
            halves.append(spec.bbox)
        self.part = Actor.merge(part_objs, name="edge_box")
        self.add_to_state_dict_registry(self.part)
        self.belt.register_rider(
            self.part,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=device),
        )
        self._part_half = torch.as_tensor(np.array(halves, dtype=np.float32), device=device)
        self._rest_half_z = self._part_half[:, 2]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox + self._sample_spawn_offset_x(env_idx, b)
        pos[:, 1] = self._sample_spawn_y(env_idx, b)
        pos[:, 2] = deck_z + self._rest_half_z[env_idx] + 0.001
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        self.part.set_pose(Pose.create_from_pq(pos, q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.part.set_linear_velocity(vel)
        self.part.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_recenter_trackers(env_idx, q)

    def _sample_spawn_offset_x(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        rcfg = self.randomization
        device = self.device
        half_x = self._part_half[env_idx, 0]

        if rcfg.overhang_frac:
            drawn = self._episode_rng.uniform(*rcfg.overhang_frac_range, size=b).astype(np.float32)
            frac = torch.as_tensor(drawn, device=device)
        else:
            frac = torch.full((b,), self._overhang_frac_default, device=device)

        if rcfg.overhang_side:
            drawn_side = self._episode_rng.choice([-1.0, 1.0], size=b).astype(np.float32)
            side = torch.as_tensor(drawn_side, device=device)
        else:
            side = torch.full((b,), self._overhang_side_default, device=device)

        return side * (self._band_half - half_x * (1.0 - 2.0 * frac))

    def _sample_spawn_y(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        oy = self.workspace_offset[1]
        start = self._conveyor_half_length - self._part_half[env_idx, 1] - self._spawn_end_margin
        return oy - start * self.belt.direction_sign[env_idx]

    def _reset_recenter_trackers(self, env_idx: torch.Tensor, spawn_q: torch.Tensor) -> None:
        if not hasattr(self, "_recenter_sustain_counter"):
            self._recenter_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._left_deck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._spawn_up = torch.zeros((self.num_envs, 3), device=self.device)
        self._recenter_sustain_counter[env_idx] = 0
        self._left_deck[env_idx] = False
        self._spawn_up[env_idx] = self._body_up(spawn_q)

    @staticmethod
    def _body_up(q: torch.Tensor) -> torch.Tensor:
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack(
            [2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
            dim=-1,
        )

    def _cross_belt_extent(self, q: torch.Tensor) -> torch.Tensor:
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        row_x = torch.stack(
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            dim=-1,
        )
        return (row_x.abs() * self._part_half).sum(dim=-1)

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        part_pos = self.part.pose.p
        part_q = self.part.pose.q
        part_vel = self.part.linear_velocity
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        self._left_deck = self._left_deck | (part_pos[:, 2] < deck_z - self._off_deck_drop)

        offset_x = torch.abs(part_pos[:, 0] - ox)
        fully_on = (offset_x + self._cross_belt_extent(part_q)) <= self._band_half
        centred = offset_x <= self._center_tol
        resting = torch.abs(part_pos[:, 2] - (deck_z + self._rest_half_z)) <= self._deck_rest_tol
        untipped = (self._body_up(part_q) * self._spawn_up).sum(dim=-1) >= math.cos(self._tip_max_tilt)
        comoving = (torch.abs(part_vel[:, 1] - self.belt.velocity) <= self._comove_tol) & (
            torch.abs(part_vel[:, 0]) <= self._lateral_speed_tol
        )
        gripper_open = self.gripper_aperture() >= self._released_aperture
        tcp_clear = torch.linalg.norm(tcp_pos - part_pos, dim=-1) >= self._released_distance
        released = gripper_open & tcp_clear
        runway = (self._conveyor_half_length - (part_pos[:, 1] - oy) * self.belt.direction_sign) >= self._runway_min

        holding = fully_on & centred & resting & untipped & released & comoving & runway & ~self._left_deck
        self._recenter_sustain_counter = self._advance_sustain(
            "recenter_sustain", self._recenter_sustain_counter, holding
        )
        success = self._recenter_sustain_counter >= self._recenter_sustain_steps

        return {
            "success": success,
            "fully_on": fully_on,
            "centred": centred,
            "resting": resting,
            "untipped": untipped,
            "gripper_open": gripper_open,
            "tcp_clear": tcp_clear,
            "comoving": comoving,
            "runway": runway,
            "left_deck": self._left_deck,
            "offset_x": offset_x,
        }

    def compute_dense_reward(self, obs, action, info):
        ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - part_pos, dim=-1))

        to_band = torch.stack(
            [part_pos[:, 0] - ox, part_pos[:, 2] - (deck_z + self._rest_half_z)],
            dim=-1,
        )
        seat_reward = 1 - torch.tanh(8 * torch.linalg.norm(to_band, dim=-1))

        riding = (info["fully_on"] & info["centred"] & info["resting"] & info["comoving"]).float()
        return reach_reward + seat_reward + 2.0 * riding

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
