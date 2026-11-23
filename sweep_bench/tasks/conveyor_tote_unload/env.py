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
from sweep_bench.tasks.conveyor_base.on_belt import belt_carries
from sweep_bench.tasks.conveyor_tote_unload.assets import build_tote
from sweep_bench.tasks.conveyor_tote_unload.config import RandomizationConfig
from sweep_bench.tasks.conveyor_tote_unload.scripted import ScriptedToteUnloadMixin
from sweep_bench.tasks.conveyor_tote_unload.variants import TOTE_UNLOAD_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorToteUnload-v0")
@register_env("ConveyorToteUnload-v0", max_episode_steps=300)
class ConveyorToteUnloadEnv(ScriptedToteUnloadMixin, ConveyorWorkspaceEnv):
    """Extract the part from an open-top tote riding the belt, without disturbing the tote."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(TOTE_UNLOAD_OBJECT_IDS)

    _conveyor_half_width: float = 0.12

    _tote_inner_half_x: float = 0.075
    _tote_inner_half_y: float = 0.075
    _tote_wall_half_thickness: float = 0.005
    _tote_wall_half_height: float = 0.020
    _tote_floor_half_height: float = 0.004
    _tote_lip: float = 0.006
    _tote_patch_radius: float = 0.06
    _part_grasp_width_fallback: float = 0.06

    _grasp_width_tol: float = 0.008
    _carry_speed_match_threshold: float = 0.03
    _lift_clearance: float = 0.10
    _unload_sustain_steps: int = 8

    _carrier_z_tol: float = 0.006
    _carrier_x_tol: float = 0.04
    _carrier_tilt_deg: float = 10.0
    _carrier_track_tol: float = 0.08

    default_prompt: str = "take the part out of the tote on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _tote_outer_half_y(self) -> float:
        return self._tote_inner_half_y + 2.0 * self._tote_wall_half_thickness

    @property
    def _tote_floor_local_z(self) -> float:
        return self._tote_floor_half_height

    @property
    def _tote_rim_local_z(self) -> float:
        return self._tote_floor_half_height + 2.0 * self._tote_wall_half_height

    @property
    def _tote_mouth_local_z(self) -> float:
        return self._tote_rim_local_z + self._tote_lip

    @property
    def _tote_rest_z(self) -> float:
        return self.workspace_offset[2] + self._conveyor_deck_height + self._tote_floor_half_height

    def _to_world(self, tote_pose: Pose, local: torch.Tensor) -> torch.Tensor:
        return (tote_pose * Pose.create_from_pq(local)).p

    def _in_tote_frame(self, world: torch.Tensor) -> torch.Tensor:
        return (self.tote.pose.inv() * Pose.create_from_pq(world)).p

    def _tote_local_world_z(self, local_z: float) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 2] = local_z
        return self._to_world(self.tote.pose, local)[:, 2]

    def tote_rim_world_z(self) -> torch.Tensor:
        return self._tote_local_world_z(self._tote_rim_local_z)

    def tote_mouth_world_z(self) -> torch.Tensor:
        return self._tote_local_world_z(self._tote_mouth_local_z)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        tote_builder = build_tote(
            self.scene,
            inner_half_x=self._tote_inner_half_x,
            inner_half_y=self._tote_inner_half_y,
            wall_half_thickness=self._tote_wall_half_thickness,
            wall_half_height=self._tote_wall_half_height,
            floor_half_height=self._tote_floor_half_height,
            lip=self._tote_lip,
        )
        tote_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._tote_floor_half_height + 0.001])
        self.tote = tote_builder.build(name="tote")
        self.belt.register_rider(self.tote, patch_radius=self._tote_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        specs = [spec_by_id[name] for name in self._part_names]

        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.50])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"tote_part-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            obj.angular_damping = 5.0
            objs.append(obj)
        self.part = Actor.merge(objs, name="tote_part")
        self.add_to_state_dict_registry(self.part)

        def _tensor(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        self._part_half_z = _tensor([s.bbox[2] for s in specs])
        self._part_half_fp = _tensor([math.hypot(s.bbox[0], s.bbox[1]) for s in specs])
        self._part_grasp_width = _tensor(
            [s.grasp_width if s.grasp_width is not None else self._part_grasp_width_fallback for s in specs]
        )

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"take the {name} out of the tote on the conveyor belt" for name in self._part_display]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        tote_pos = self._sample_tote_position(env_idx, b)
        tote_q = self._sample_tote_orientation(b)
        tote_pose = Pose.create_from_pq(tote_pos, tote_q)
        self.tote.set_pose(tote_pose)
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)
        self.tote.set_linear_velocity(vel)
        self.tote.set_angular_velocity(zero)

        local = torch.zeros((b, 3), device=device)
        local[:, :2] = self._sample_part_offset(b)
        local[:, 2] = self._tote_floor_local_z + self._part_half_z[env_idx] + 0.0005
        part_q = torch.zeros((b, 4), device=device)
        part_q[:, 0] = 1.0
        self.part.set_pose(Pose.create_from_pq(self._to_world(tote_pose, local), part_q))
        self.part.set_linear_velocity(vel)
        self.part.set_angular_velocity(zero)

        self._reset_unload_trackers(env_idx)

    def _sample_tote_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.carrier_yaw:
            lo, hi = self.randomization.carrier_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_part_offset(self, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.part_offset:
            return torch.zeros((b, 2), device=device)
        lo, hi = self.randomization.part_offset_range
        draws = self._episode_rng.uniform(lo, hi, size=(b, 2)).astype(np.float32)
        return torch.as_tensor(draws, device=device)

    def _sample_tote_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._tote_outer_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = self._tote_rest_z + 0.001
        return pos

    def _reset_unload_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_unload_sustain"):
            self._unload_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._prev_part_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._unload_sustain[env_idx] = 0
        self._prev_part_pos[env_idx] = self.part.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]

    def evaluate(self):
        ox, oy, _oz = self.workspace_offset
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tote_pose = self.tote.pose
        tote_pos = tote_pose.p
        part_pos = self.part.pose.p
        belt_v = self.belt.velocity

        finger_width = self.gripper_aperture()
        width_matches = torch.abs(finger_width - self._part_grasp_width) <= self._grasp_width_tol
        rel_vel = ((part_pos - tcp_pos) - (self._prev_part_pos - self._prev_tcp_pos)) * self.control_freq
        self._prev_part_pos = part_pos.clone()
        self._prev_tcp_pos = tcp_pos.clone()
        carried = torch.linalg.norm(rel_vel, dim=-1) < self._carry_speed_match_threshold
        part_held = width_matches & agent.is_grasping(self.part) & carried

        part_lifted = part_pos[:, 2] >= self._tote_rest_z + self._tote_rim_local_z + self._lift_clearance

        local = self._in_tote_frame(part_pos)
        over_mouth = (torch.abs(local[:, 0]) <= self._tote_inner_half_x + self._part_half_fp) & (
            torch.abs(local[:, 1]) <= self._tote_inner_half_y + self._part_half_fp
        )
        below_rim = (local[:, 2] - self._part_half_z) <= self._tote_rim_local_z
        part_extracted = ~(over_mouth & below_rim)

        carrier_z_ok = torch.abs(tote_pos[:, 2] - self._tote_rest_z) <= self._carrier_z_tol
        carrier_x_ok = torch.abs(tote_pos[:, 0] - ox) <= self._carrier_x_tol
        carrier_upright = quaternion_to_matrix(tote_pose.q)[:, 2, 2] >= math.cos(math.radians(self._carrier_tilt_deg))
        carrier_on_belt = torch.abs(tote_pos[:, 1] - oy) <= self._conveyor_half_length - self._tote_outer_half_y
        tote_v_y = self.tote.linear_velocity[:, 1]
        carrier_tracking = torch.abs(tote_v_y - belt_v) <= self._carrier_track_tol
        carrier_travelling = belt_carries(tote_v_y, belt_v, 0.5)
        carrier_ok = (
            carrier_z_ok & carrier_x_ok & carrier_upright & carrier_on_belt & carrier_tracking & carrier_travelling
        )

        unloaded = part_held & part_lifted & carrier_ok

        self._unload_sustain = self._advance_sustain("unload_sustain", self._unload_sustain, unloaded)
        success = self._unload_sustain >= self._unload_sustain_steps

        return {
            "success": success,
            "part_held": part_held,
            "part_lifted": part_lifted,
            "part_extracted": part_extracted,
            "carrier_ok": carrier_ok,
            "carrier_z_ok": carrier_z_ok,
            "carrier_x_ok": carrier_x_ok,
            "carrier_upright": carrier_upright,
            "carrier_on_belt": carrier_on_belt,
            "carrier_tracking": carrier_tracking,
            "carrier_travelling": carrier_travelling,
            "unload_sustain": self._unload_sustain,
            "part_above_rim": part_pos[:, 2] - self.tote_rim_world_z(),
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - part_pos, dim=-1))
        above_rim = part_pos[:, 2] - self.tote_rim_world_z()
        lift_reward = torch.clamp(above_rim / self._lift_clearance, 0, 1)
        return reach_reward + 2.0 * lift_reward * info["part_held"].float() * info["carrier_ok"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
