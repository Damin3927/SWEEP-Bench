from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_edge_flush.assets import build_rail_tray
from sweep_bench.tasks.conveyor_edge_flush.config import RandomizationConfig
from sweep_bench.tasks.conveyor_edge_flush.scripted import ScriptedEdgeFlushMixin
from sweep_bench.tasks.conveyor_edge_flush.variants import EDGE_FLUSH_OBJECT_IDS, SLAB_SPEC
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec

_MIN_SLAB_ASPECT = 1.3


@bench_task("ConveyorEdgeFlush-v0")
@register_env("ConveyorEdgeFlush-v0", max_episode_steps=300)
class ConveyorEdgeFlushEnv(ScriptedEdgeFlushMixin, ConveyorWorkspaceEnv):
    """Square a skewed slab up against the rail of the tray it rides on."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(EDGE_FLUSH_OBJECT_IDS, extra=(SLAB_SPEC,))

    _tray_half_x: float = 0.0625
    _tray_half_y: float = 0.080
    _tray_floor_thickness: float = 0.006
    _rail_thickness: float = 0.008
    _rail_height: float = 0.005
    _tray_density: float = 4000.0
    _tray_static_friction: float = 0.4
    _tray_dynamic_friction: float = 0.3
    _tray_restitution: float = 0.0
    _tray_spawn_gap: float = 0.001
    _tray_patch_radius: float = 0.05

    _slab_spawn_offset: float = 0.035
    _slab_spawn_yaw_deg: float = 15.0
    _slab_spawn_gap: float = 0.001

    _flat_tol: float = 0.004
    _withdraw_dist: float = 0.06
    _comoving_rel_speed_floor: float = 0.02
    _comoving_rel_speed_slope: float = 0.04
    _flush_sustain_steps: int = 8
    _tray_lift_margin: float = 0.006
    _tray_drag_margin: float = 0.04

    default_prompt: str = "push the slab flush against the tray's rail on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        flush_tol: float = 0.006,
        square_tol_deg: float = 3.0,
        **kwargs,
    ) -> None:
        for spec in self.OBJECT_CATALOG:
            self._validate_slab_spec(spec)
        if flush_tol <= 0.0 or square_tol_deg <= 0.0:
            raise ValueError(f"flush_tol/square_tol_deg must be positive, got ({flush_tol}, {square_tol_deg})")
        min_half_along = min(max(spec.bbox[0], spec.bbox[1]) for spec in self.OBJECT_CATALOG)
        entailed_deg = math.degrees(math.asin(min(1.0, flush_tol / min_half_along)))
        if entailed_deg <= square_tol_deg:
            raise ValueError(
                f"flush_tol={flush_tol} already bounds the yaw of the narrowest adopted slab to "
                f"{entailed_deg:.1f} deg, which entails square_tol_deg={square_tol_deg}: the yaw term "
                f"could never fail on its own. Tighten flush_tol or loosen square_tol_deg."
            )
        self.flush_tol: float = flush_tol
        self.square_tol_deg: float = square_tol_deg
        super().__init__(*args, **kwargs)

    @classmethod
    def _validate_slab_spec(cls, spec: ObjectSpec) -> None:
        hx, hy, hz = spec.bbox
        if hz >= min(hx, hy):
            raise ValueError(
                f"slab {spec.id!r} bbox={spec.bbox} is not flat: its vertical half-extent must be the smallest"
            )
        aspect = max(hx, hy) / min(hx, hy)
        if aspect < _MIN_SLAB_ASPECT:
            raise ValueError(
                f"slab {spec.id!r} bbox={spec.bbox} has aspect {aspect:.2f} < {_MIN_SLAB_ASPECT}: "
                f"its long edge is ambiguous, so the yaw term has no well-defined datum"
            )

    @property
    def _rail_inner_x(self) -> float:
        return self._tray_half_x - self._rail_thickness

    @property
    def _square_tol_rad(self) -> float:
        return math.radians(self.square_tol_deg)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        tray_builder = build_rail_tray(
            self.scene,
            half_x=self._tray_half_x,
            half_y=self._tray_half_y,
            floor_thickness=self._tray_floor_thickness,
            rail_thickness=self._rail_thickness,
            rail_height=self._rail_height,
            density=self._tray_density,
            static_friction=self._tray_static_friction,
            dynamic_friction=self._tray_dynamic_friction,
            restitution=self._tray_restitution,
        )
        tray_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._tray_rest_z_above_deck])
        self.tray = tray_builder.build(name="rail_tray")
        self.belt.register_rider(self.tray, patch_radius=self._tray_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._slab_names: list[str] = [str(name) for name in names]
        self._slab_display: list[str] = [display_name(spec_by_id[name]) for name in self._slab_names]
        slab_objs: list[Actor] = []
        half_across: list[float] = []
        half_along: list[float] = []
        half_thick: list[float] = []
        long_is_x: list[bool] = []
        for i, name in enumerate(self._slab_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.1 + 0.01 * i])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            slab_objs.append(obj)
            hx, hy, hz = spec.bbox
            half_across.append(min(hx, hy))
            half_along.append(max(hx, hy))
            half_thick.append(hz)
            long_is_x.append(hx > hy)
        self.slab = Actor.merge(slab_objs, name="flush_slab")
        self.add_to_state_dict_registry(self.slab)
        device = self.device
        self._slab_half_across = torch.as_tensor(half_across, dtype=torch.float32, device=device)
        self._slab_half_along = torch.as_tensor(half_along, dtype=torch.float32, device=device)
        self._slab_half_thick = torch.as_tensor(half_thick, dtype=torch.float32, device=device)
        self._slab_long_is_x = torch.as_tensor(long_is_x, dtype=torch.bool, device=device)

    @property
    def _tray_rest_z_above_deck(self) -> float:
        return self._tray_floor_thickness + self._tray_spawn_gap

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        oz = self.workspace_offset[2]
        deck_z = oz + self._conveyor_deck_height

        anchor = self._sample_tray_position(env_idx, b)

        tray_pos = torch.zeros((b, 3), device=device)
        tray_pos[:, 0] = anchor[:, 0]
        tray_pos[:, 1] = anchor[:, 1]
        tray_pos[:, 2] = deck_z + self._tray_rest_z_above_deck
        tray_q = torch.zeros((b, 4), device=device)
        tray_q[:, 0] = 1.0
        if self.randomization.rail_side:
            flips = self._episode_rng.choice(np.array([False, True]), size=b)
            mirrored = torch.as_tensor(np.asarray(flips, dtype=bool), device=device)
            tray_q[:, 0] = torch.where(mirrored, torch.zeros_like(tray_q[:, 0]), tray_q[:, 0])
            tray_q[:, 3] = torch.where(mirrored, torch.ones_like(tray_q[:, 3]), tray_q[:, 3])
        tray_pose = Pose.create_from_pq(tray_pos, tray_q)
        self.tray.set_pose(tray_pose)
        tray_vel = torch.zeros((b, 3), device=device)
        tray_vel[:, 1] = self.belt.velocity[env_idx]
        self.tray.set_linear_velocity(tray_vel)
        self.tray.set_angular_velocity(torch.zeros((b, 3), device=device))

        offset, skew_yaw = self._slab_spawn_offsets(b)
        local_pos = torch.zeros((b, 3), device=device)
        local_pos[:, 0] = self._rail_inner_x - self._slab_half_across[env_idx] - offset
        local_pos[:, 1] = 0.0
        local_pos[:, 2] = self._slab_half_thick[env_idx] + self._slab_spawn_gap
        base_yaw = torch.where(
            self._slab_long_is_x[env_idx],
            torch.full((b,), math.pi / 2, device=device),
            torch.zeros((b,), device=device),
        )
        yaw = base_yaw + skew_yaw
        local_q = torch.zeros((b, 4), device=device)
        local_q[:, 0] = torch.cos(yaw * 0.5)
        local_q[:, 3] = torch.sin(yaw * 0.5)
        self.slab.set_pose(tray_pose * Pose.create_from_pq(local_pos, local_q))
        slab_vel = torch.zeros((b, 3), device=device)
        slab_vel[:, 1] = self.belt.velocity[env_idx]
        self.slab.set_linear_velocity(slab_vel)
        self.slab.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_flush_trackers(env_idx)

    def _sample_tray_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._tray_half_y - 0.04
        anchor = torch.zeros((b, 2), device=device)
        anchor[:, 0] = ox
        anchor[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        return anchor

    def _slab_spawn_offsets(self, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        rcfg = self.randomization
        if not rcfg.initial_skew:
            offset = torch.full((b,), self._slab_spawn_offset, device=device)
            yaw = torch.full((b,), math.radians(self._slab_spawn_yaw_deg), device=device)
            return offset, yaw
        off_np = self._episode_rng.uniform(*rcfg.skew_offset_range, size=b).astype(np.float32)
        yaw_np = np.radians(self._episode_rng.uniform(*rcfg.skew_yaw_range_deg, size=b)).astype(np.float32)
        return torch.as_tensor(off_np, device=device), torch.as_tensor(yaw_np, device=device)

    @property
    def episode_objects(self) -> list[str]:
        return list(self._slab_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"push the {name} flush against the tray's rail on the conveyor belt" for name in self._slab_display]

    @property
    def _comoving_rel_speed(self) -> torch.Tensor:
        return self._comoving_rel_speed_floor + self._comoving_rel_speed_slope * tolerance_speed(self.belt.velocity)

    def _reset_flush_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_flush_sustain_counter"):
            self._flush_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._flush_sustain_counter[env_idx] = 0

    def _slab_datum_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = self.tray.pose.inv() * self.slab.pose
        rel_p = rel.p
        rot = quaternion_to_matrix(rel.q)
        body_x, body_y, body_z = rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]

        long_is_x = self._slab_long_is_x.unsqueeze(-1)
        long_axis = torch.where(long_is_x, body_x, body_y)
        short_axis = torch.where(long_is_x, body_y, body_x)

        short_axis = torch.where(short_axis[:, 0:1] >= 0.0, short_axis, -short_axis)
        long_axis = torch.where(long_axis[:, 1:2] >= 0.0, long_axis, -long_axis)
        up_axis = torch.where(body_z[:, 2:3] >= 0.0, body_z, -body_z)

        half_across = self._slab_half_across.unsqueeze(-1)
        edge_mid = rel_p + half_across * short_axis
        edge_gap = self._rail_inner_x - edge_mid[:, 0]
        yaw = torch.atan2(long_axis[:, 0], long_axis[:, 1])
        bottom_edge_z = edge_mid[:, 2] - self._slab_half_thick * up_axis[:, 2]
        return edge_gap, yaw, bottom_edge_z

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        edge_gap, yaw, bottom_edge_z = self._slab_datum_state()

        flush = torch.abs(edge_gap) < self.flush_tol
        square = torch.abs(yaw) < self._square_tol_rad
        flat = torch.abs(bottom_edge_z) < self._flat_tol

        agent = cast("Panda", self.agent)
        tcp_to_slab_xy = torch.linalg.norm(agent.tcp.pose.p[:, :2] - self.slab.pose.p[:, :2], dim=-1)
        withdrawn = tcp_to_slab_xy > self._withdraw_dist

        rel_speed = torch.linalg.norm(self.slab.linear_velocity - self.tray.linear_velocity, dim=-1)
        comoving = rel_speed < self._comoving_rel_speed

        tray_pos = self.tray.pose.p
        tray_on_belt = (
            (torch.abs(tray_pos[:, 2] - (deck_z + self._tray_rest_z_above_deck)) <= self._tray_lift_margin)
            & (torch.abs(tray_pos[:, 0] - ox) <= self._tray_drag_margin)
            & (torch.abs(tray_pos[:, 1] - oy) < self._conveyor_half_length)
        )

        holding = flush & square & flat & withdrawn & comoving & tray_on_belt
        self._flush_sustain_counter = self._advance_sustain("flush_sustain", self._flush_sustain_counter, holding)
        success = self._flush_sustain_counter >= self._flush_sustain_steps

        return {
            "success": success,
            "flush": flush,
            "square": square,
            "flat": flat,
            "withdrawn": withdrawn,
            "comoving": comoving,
            "tray_on_belt": tray_on_belt,
            "edge_gap": edge_gap,
            "slab_yaw": yaw,
            "slab_bottom_edge_z": bottom_edge_z,
            "slab_tray_rel_speed": rel_speed,
            "flush_sustain": self._flush_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        slab_pos = self.slab.pose.p

        tcp_to_slab = torch.linalg.norm(tcp_pos - slab_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_slab)

        offset_progress = 1 - torch.tanh(8 * info["edge_gap"].abs())
        square_progress = 1 - torch.tanh(4 * info["slab_yaw"].abs())

        return reach_reward + offset_progress + square_progress + 2.0 * (info["flush"] & info["square"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
