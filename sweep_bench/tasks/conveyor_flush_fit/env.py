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
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_flush_fit.assets import build_dual_well_carrier
from sweep_bench.tasks.conveyor_flush_fit.config import RandomizationConfig
from sweep_bench.tasks.conveyor_flush_fit.scripted import ScriptedFlushFitMixin
from sweep_bench.tasks.conveyor_flush_fit.variants import FLUSH_FIT_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec


@bench_task("ConveyorFlushFit-v0")
@register_env("ConveyorFlushFit-v0", max_episode_steps=300)
class ConveyorFlushFitEnv(ScriptedFlushFitMixin, ConveyorWorkspaceEnv):
    """Drop the held part into the well whose DEPTH makes it sit flush."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(FLUSH_FIT_OBJECT_IDS)

    _well_inner_half: float = 0.036
    _wall_half_thickness: float = 0.005
    _base_half_height: float = 0.006
    _well_chamfer: float = 0.004
    _well_footprint_clearance: float = 0.015

    _match_clearance: float = 0.002
    _decoy_depth_offset: float = 0.025

    _MATCH_WELL: int = 0
    _DECOY_WELL: int = 1

    _carrier_patch_radius: float = 0.040
    _part_patch_radius: float = 0.015

    _flush_tol: float = 0.005
    _bottomed_tol: float = 0.004
    _well_inset: float = 0.004
    _face_seat_cos: float = math.cos(math.radians(10.0))
    _released_finger_width: float = 0.045
    _released_distance_threshold: float = 0.08
    _comove_tol_base: float = 0.05
    _comove_tol_speed_gain: float = 0.06
    _carrier_comove_tol_base: float = 0.08
    _carrier_comove_tol_speed_gain: float = 0.06
    _carrier_x_tol: float = 0.04
    _carrier_tilt_cos: float = math.cos(math.radians(10.0))
    _carrier_y_tol: float = 0.88
    _flush_sustain_steps: int = 10

    default_prompt: str = "seat the part in the well on the conveyor carrier where its top sits flush"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        objects: tuple[str, ...] | None = None,
        **kwargs,
    ) -> None:
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(*args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, objects=objects, **kwargs)

    @property
    def _well_pitch(self) -> float:
        return 2.0 * (self._well_inner_half + self._wall_half_thickness)

    @property
    def _carrier_outer_half_y(self) -> float:
        return 2.0 * self._well_inner_half + 3.0 * self._wall_half_thickness

    def _well_local_y(self, well: int) -> float:
        return (well - 0.5) * self._well_pitch

    def _well_local_offset(self, well: int) -> torch.Tensor:
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 1] = self._well_local_y(well)
        return offset

    def _to_world(self, carrier_pose: Pose, local: torch.Tensor) -> torch.Tensor:
        return (carrier_pose * Pose.create_from_pq(local)).p

    def _in_carrier_frame(self, world: torch.Tensor) -> torch.Tensor:
        return (self.carrier.pose.inv() * Pose.create_from_pq(world)).p

    def target_well_offset_local(self) -> torch.Tensor:
        return self._well_local_offset(self._MATCH_WELL)

    def target_well_world_pos(self) -> torch.Tensor:
        return self._to_world(self.carrier.pose, self.target_well_offset_local())

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        specs = [spec_by_id[name] for name in self._part_names]

        signs = self._batched_episode_rng.choice(np.array([1.0, -1.0]), replace=True)
        depth_pairs: list[tuple[float, float]] = []
        for spec, sign in zip(specs, signs, strict=True):
            footprint = max(spec.bbox[0], spec.bbox[1])
            if footprint > self._well_inner_half - self._well_footprint_clearance:
                raise ValueError(
                    f"{spec.id}: horizontal half-extent {footprint:.4f} m leaves under "
                    f"{self._well_footprint_clearance:.3f} m of clearance in a "
                    f"{self._well_inner_half:.3f} m half-width well"
                )
            match_depth = 2.0 * spec.bbox[2] + self._match_clearance
            decoy_depth = match_depth + float(sign) * self._decoy_depth_offset
            pair = [0.0, 0.0]
            pair[self._MATCH_WELL] = match_depth
            pair[self._DECOY_WELL] = decoy_depth
            depth_pairs.append((pair[0], pair[1]))

        self.carrier = self._build_carrier_group(depth_pairs, deck_z + 0.30)
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        self.part = self._build_part_group(specs, "flush_part", deck_z + 0.50)
        self.belt.register_rider(
            self.part,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._part_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        depths = torch.as_tensor(depth_pairs, dtype=torch.float32, device=self.device)
        wall_height = depths.max(dim=1).values
        self._well_depth = depths
        self._carrier_top_local_z = self._base_half_height + wall_height
        self._carrier_mouth_local_z = self._carrier_top_local_z + self._well_chamfer
        self._well_floor_local_z = self._carrier_top_local_z.unsqueeze(-1) - depths
        self._match_is_deeper = depths[:, self._MATCH_WELL] > depths[:, self._DECOY_WELL]

        self._part_bbox = torch.as_tensor([s.bbox for s in specs], dtype=torch.float32, device=self.device)
        self._part_half_z = self._part_bbox[:, 2]
        self._part_grasp_half = torch.as_tensor(
            [(s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1])) / 2.0 for s in specs],
            dtype=torch.float32,
            device=self.device,
        )

    def _build_carrier_group(self, depth_pairs: list[tuple[float, float]], park_z: float) -> Actor:
        ox, oy, _oz = self.workspace_offset
        objs: list[Actor] = []
        for i, depths in enumerate(depth_pairs):
            builder = build_dual_well_carrier(
                self.scene,
                depths=depths,
                well_inner_half=self._well_inner_half,
                wall_half_thickness=self._wall_half_thickness,
                base_half_height=self._base_half_height,
                chamfer=self._well_chamfer,
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy, park_z])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"flush_carrier-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        merged = Actor.merge(objs, name="flush_carrier")
        self.add_to_state_dict_registry(merged)
        return merged

    def _build_part_group(self, specs: list[ObjectSpec], group: str, park_z: float) -> Actor:
        ox, oy, _oz = self.workspace_offset
        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, park_z])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{group}-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        merged = Actor.merge(objs, name=group)
        self.add_to_state_dict_registry(merged)
        return merged

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"seat the {name} in the well on the conveyor carrier where its top sits flush"
            for name in self._part_display
        ]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = self._part_grasp_half[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_flush_trackers(env_idx)

        carrier_pos = self._sample_carrier_position(env_idx, b)
        carrier_q = self._sample_carrier_orientation(env_idx, b)
        carrier_pose = Pose.create_from_pq(carrier_pos, carrier_q)
        self.carrier.set_pose(carrier_pose)
        carrier_vel = torch.zeros((b, 3), device=device)
        carrier_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(carrier_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        part_q = torch.zeros((b, 4), device=device)
        part_q[:, 0] = 1.0
        self.part.set_pose(Pose.create_from_pq(tcp_pos, part_q))
        self.part.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.part.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _sample_carrier_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        flip = self._episode_rng.choice(np.array([0.0, 1.0]), size=b)
        yaw = torch.as_tensor((flip * math.pi).astype(np.float32), device=device)
        self._match_well_flipped[env_idx] = torch.as_tensor(flip.astype(bool), device=device)
        if self.randomization.carrier_yaw:
            lo, hi = self.randomization.carrier_yaw_range
            jitter = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            yaw = yaw + torch.as_tensor(jitter, device=device)
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(yaw * 0.5)
        q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_outer_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._base_half_height + 0.001
        return pos

    def _reset_flush_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_flush_sustain_counter"):
            self._flush_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._match_well_flipped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._flush_sustain_counter[env_idx] = 0

    def evaluate(self):
        agent = cast("Panda", self.agent)
        carrier_pose = self.carrier.pose
        carrier_pos = carrier_pose.p
        part_pos = self.part.pose.p
        belt_v = self.belt.velocity
        tol_speed = tolerance_speed(belt_v)

        part_local = self._in_carrier_frame(part_pos)

        near_well = (part_local[:, 1] > 0.0).long()
        near_well_y = (near_well.to(torch.float32) - 0.5) * self._well_pitch
        near_floor_z = torch.gather(self._well_floor_local_z, 1, near_well.unsqueeze(-1)).squeeze(-1)

        reach = self._well_inner_half - self._well_inset
        in_well = (torch.abs(part_local[:, 0]) <= reach) & (torch.abs(part_local[:, 1] - near_well_y) <= reach)

        up = torch.zeros((self.num_envs, 3), device=self.device)
        up[:, 2] = 1.0
        carrier_axis = self._to_world(carrier_pose, up) - carrier_pos
        support_half_z, axis_alignment = self._support_half_z(carrier_axis)

        bottomed = torch.abs((part_local[:, 2] - support_half_z) - near_floor_z) <= self._bottomed_tol

        flush_error = (part_local[:, 2] + support_half_z) - self._carrier_top_local_z
        flush = torch.abs(flush_error) <= self._flush_tol

        face_seated = axis_alignment >= self._face_seat_cos

        finger_width = self.gripper_aperture()
        released = (finger_width >= self._released_finger_width) & (
            torch.linalg.norm(agent.tcp.pose.p - part_pos, dim=-1) > self._released_distance_threshold
        )

        rel_v = self.part.linear_velocity - self.carrier.linear_velocity
        comove_tol = self._comove_tol_base + self._comove_tol_speed_gain * tol_speed
        comoving = torch.linalg.norm(rel_v, dim=-1) < comove_tol

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        tilt_cos = carrier_axis[:, 2]
        carrier_comove_tol = self._carrier_comove_tol_base + self._carrier_comove_tol_speed_gain * tol_speed
        carrier_on_belt = (
            (torch.abs(carrier_pos[:, 0] - ox) <= self._carrier_x_tol)
            & (torch.abs(self.carrier.linear_velocity[:, 1] - belt_v) <= carrier_comove_tol)
            & (tilt_cos >= self._carrier_tilt_cos)
            & (carrier_pos[:, 2] > deck_z)
            & (torch.abs(carrier_pos[:, 1] - oy) <= self._carrier_y_tol)
        )

        seated = in_well & bottomed & flush & face_seated & released & comoving & carrier_on_belt

        self._flush_sustain_counter = self._advance_sustain("flush_sustain", self._flush_sustain_counter, seated)
        success = self._flush_sustain_counter >= self._flush_sustain_steps

        return {
            "success": success,
            "in_well": in_well,
            "bottomed": bottomed,
            "flush": flush,
            "face_seated": face_seated,
            "released": released,
            "comoving": comoving,
            "carrier_on_belt": carrier_on_belt,
            "flush_sustain": self._flush_sustain_counter,
            "near_well": near_well,
            "flush_error": flush_error,
            "match_is_deeper": self._match_is_deeper,
            "match_well_flipped": self._match_well_flipped,
        }

    def _support_half_z(self, carrier_axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        axis_in_part = (self.part.pose.inv() * Pose.create_from_pq(self.part.pose.p + carrier_axis)).p.abs()
        return (axis_in_part * self._part_bbox).sum(dim=-1), axis_in_part.max(dim=-1).values

    def compute_dense_reward(self, obs, action, info):
        part_pos = self.part.pose.p
        well_pos = self.target_well_world_pos()

        xy_dist = torch.linalg.norm(part_pos[:, :2] - well_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        fitted = (info["in_well"] & info["bottomed"] & info["flush"] & info["face_seated"]).float()
        return carry_reward + 2.0 * fitted

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
