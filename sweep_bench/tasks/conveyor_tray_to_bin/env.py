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

from sweep_bench.assets.procedural import build_bin
from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import belt_carries
from sweep_bench.tasks.conveyor_tray_to_bin.config import RandomizationConfig
from sweep_bench.tasks.conveyor_tray_to_bin.scripted import ScriptedTrayToBinMixin
from sweep_bench.tasks.conveyor_tray_to_bin.variants import TRAY_TO_BIN_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_TRAY_COLOR = (0.72, 0.68, 0.58, 1.0)
_BIN_COLOR = (0.32, 0.36, 0.44, 1.0)


@bench_task("ConveyorTrayToBin-v0")
@register_env("ConveyorTrayToBin-v0", max_episode_steps=300)
class ConveyorTrayToBinEnv(ScriptedTrayToBinMixin, ConveyorWorkspaceEnv):
    """Lift the part out of the leading tray and drop it into the bin riding behind."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(TRAY_TO_BIN_OBJECT_IDS)

    _tray_inner_half_x: float = 0.05
    _tray_inner_half_y: float = 0.05
    _tray_wall_thickness: float = 0.006
    _tray_wall_half_height: float = 0.006
    _tray_floor_half_height: float = 0.004

    _bin_inner_half_x: float = 0.055
    _bin_inner_half_y: float = 0.055
    _bin_wall_thickness: float = 0.007
    _bin_wall_half_height: float = 0.0175
    _bin_floor_half_height: float = 0.005

    _carrier_patch_radius: float = 0.05
    _carrier_pitch: float = 0.35
    _carrier_spawn_margin: float = 0.04

    _in_bin_xy_margin: float = 0.005
    _ride_speed_match: float = 0.05
    _release_finger_margin: float = 0.01
    _released_distance: float = 0.06
    _deliver_sustain_steps: int = 4

    _tray_z_tol: float = 0.006
    _tray_x_tol: float = 0.05
    _tray_tilt_deg: float = 10.0
    _carrier_x_tol: float = 0.05
    _deck_rest_tol: float = 0.01

    default_prompt: str = "move the part from the tray into the bin on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _tray_outer_half_y(self) -> float:
        return self._tray_inner_half_y + 2.0 * self._tray_wall_thickness

    @property
    def _tray_outer_half_x(self) -> float:
        return self._tray_inner_half_x + 2.0 * self._tray_wall_thickness

    @property
    def _bin_outer_half_y(self) -> float:
        return self._bin_inner_half_y + 2.0 * self._bin_wall_thickness

    @property
    def _bin_outer_half_x(self) -> float:
        return self._bin_inner_half_x + 2.0 * self._bin_wall_thickness

    @property
    def _tray_rest_z(self) -> float:
        return self.workspace_offset[2] + self._conveyor_deck_height + self._tray_floor_half_height

    @property
    def _bin_rest_z(self) -> float:
        return self.workspace_offset[2] + self._conveyor_deck_height + self._bin_floor_half_height

    @property
    def _tray_floor_local_z(self) -> float:
        return self._tray_floor_half_height

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        tray_builder = build_bin(
            self.scene,
            inner_half_x=self._tray_inner_half_x,
            inner_half_y=self._tray_inner_half_y,
            wall_thickness=self._tray_wall_thickness,
            wall_half_height=self._tray_wall_half_height,
            floor_half_height=self._tray_floor_half_height,
            color=_TRAY_COLOR,
        )
        tray_builder.initial_pose = sapien.Pose(p=[ox, oy - 0.2, self._tray_rest_z + 0.001])
        self.tray = tray_builder.build(name="source_tray")
        self.belt.register_rider(self.tray, patch_radius=self._carrier_patch_radius)

        bin_builder = build_bin(
            self.scene,
            inner_half_x=self._bin_inner_half_x,
            inner_half_y=self._bin_inner_half_y,
            wall_thickness=self._bin_wall_thickness,
            wall_half_height=self._bin_wall_half_height,
            floor_half_height=self._bin_floor_half_height,
            color=_BIN_COLOR,
        )
        bin_builder.initial_pose = sapien.Pose(p=[ox, oy + 0.2, self._bin_rest_z + 0.001])
        self.bin = bin_builder.build(name="destination_bin")
        self.belt.register_rider(self.bin, patch_radius=self._carrier_patch_radius)

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
            obj = builder.build(name=f"tray_part-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            obj.angular_damping = 5.0
            objs.append(obj)
        self.part = Actor.merge(objs, name="tray_part")
        self.add_to_state_dict_registry(self.part)

        def _tensor(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        self._part_half_z = _tensor([s.bbox[2] for s in specs])
        self._part_half_fp = _tensor([max(s.bbox[0], s.bbox[1]) for s in specs])
        self._part_grasp_width = _tensor(
            [s.grasp_width if s.grasp_width is not None else 2.0 * s.bbox[0] for s in specs]
        )
        self._part_half_width = 0.5 * self._part_grasp_width

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"move the {name} from the tray into the bin on the conveyor belt" for name in self._part_display]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        pitch = self._sample_pitch(b)
        pair_x, tray_y, bin_y = self._sample_pair_xy(env_idx, b, pitch)

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)

        tray_pos = torch.zeros((b, 3), device=device)
        tray_pos[:, 0] = pair_x
        tray_pos[:, 1] = tray_y
        tray_pos[:, 2] = self._tray_rest_z + 0.001
        tray_pose = Pose.create_from_pq(tray_pos, self._sample_tray_orientation(b))
        self.tray.set_pose(tray_pose)
        self.tray.set_linear_velocity(vel)
        self.tray.set_angular_velocity(zero)

        bin_pos = torch.zeros((b, 3), device=device)
        bin_pos[:, 0] = pair_x
        bin_pos[:, 1] = bin_y
        bin_pos[:, 2] = self._bin_rest_z + 0.001
        bin_q = torch.zeros((b, 4), device=device)
        bin_q[:, 0] = 1.0
        self.bin.set_pose(Pose.create_from_pq(bin_pos, bin_q))
        self.bin.set_linear_velocity(vel)
        self.bin.set_angular_velocity(zero)

        local = torch.zeros((b, 3), device=device)
        local[:, 2] = self._tray_floor_local_z + self._part_half_z[env_idx] + 0.0005
        part_q = torch.zeros((b, 4), device=device)
        part_q[:, 0] = 1.0
        self.part.set_pose(Pose.create_from_pq((tray_pose * Pose.create_from_pq(local)).p, part_q))
        self.part.set_linear_velocity(vel)
        self.part.set_angular_velocity(zero)

        self._reset_delivery_trackers(env_idx)

    def _sample_pitch(self, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.carrier_pitch:
            return torch.full((b,), self._carrier_pitch, device=device)
        lo, hi = self.randomization.carrier_pitch_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)

    def _sample_tray_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.tray_yaw:
            lo, hi = self.randomization.tray_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_pair_xy(
        self, env_idx: torch.Tensor, b: int, pitch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ox, oy, _oz = self.workspace_offset
        sign = self.belt.direction_sign[env_idx]
        start_y = (
            self._conveyor_half_length
            - max(self._tray_outer_half_y, self._bin_outer_half_y)
            - (self._carrier_spawn_margin)
        )
        trail_y = oy - start_y * sign
        pair_x = torch.full((b,), ox, device=self.device)
        return pair_x, trail_y + pitch * sign, trail_y

    def _reset_delivery_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_deliver_sustain"):
            self._deliver_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._delivered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._tray_disturbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._deck_rested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._prev_part_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_bin_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._deliver_sustain[env_idx] = 0
        self._delivered[env_idx] = False
        self._tray_disturbed[env_idx] = False
        self._deck_rested[env_idx] = False
        self._prev_part_pos[env_idx] = self.part.pose.p[env_idx]
        self._prev_bin_pos[env_idx] = self.bin.pose.p[env_idx]

    def _outside_footprint(
        self, carrier_pose: Pose, half_x: float, half_y: float, world_p: torch.Tensor
    ) -> torch.Tensor:
        local = (carrier_pose.inv() * Pose.create_from_pq(world_p)).p
        return (torch.abs(local[:, 0]) > half_x + self._part_half_fp) | (
            torch.abs(local[:, 1]) > half_y + self._part_half_fp
        )

    def _success_impossible(self, info: dict) -> torch.Tensor:
        return self._tray_disturbed | self._deck_rested | self._departed(self.bin) | self._departed(self.part)

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p
        bin_pose = self.bin.pose
        bin_pos = bin_pose.p
        tray_pose = self.tray.pose
        tray_pos = tray_pose.p
        belt_v = self.belt.velocity

        local = (bin_pose.inv() * Pose.create_from_pq(part_pos)).p
        in_bin_xy = (torch.abs(local[:, 0]) < self._bin_inner_half_x - self._part_half_fp - self._in_bin_xy_margin) & (
            torch.abs(local[:, 1]) < self._bin_inner_half_y - self._part_half_fp - self._in_bin_xy_margin
        )
        in_bin_z = (local[:, 2] > self._bin_floor_half_height) & (
            local[:, 2] < self._bin_floor_half_height + 2.0 * self._bin_wall_half_height
        )

        rel_vel = ((part_pos - bin_pos) - (self._prev_part_pos - self._prev_bin_pos)) * self.control_freq
        self._prev_part_pos = part_pos.clone()
        self._prev_bin_pos = bin_pos.clone()
        riding_with_bin = torch.linalg.norm(rel_vel, dim=-1) < self._ride_speed_match

        finger_width = self.gripper_aperture()
        fingers_open = finger_width >= self._part_grasp_width + self._release_finger_margin
        hand_clear = torch.linalg.norm(tcp_pos - part_pos, dim=-1) > self._released_distance
        released = fingers_open & hand_clear

        bin_on_belt = (torch.abs(bin_pos[:, 1] - oy) <= self._conveyor_half_length - self._bin_outer_half_y) & (
            torch.abs(bin_pos[:, 0] - ox) <= self._carrier_x_tol
        )
        bin_travelling = belt_carries(self.bin.linear_velocity[:, 1], belt_v, 0.5)

        delivering = in_bin_xy & in_bin_z & riding_with_bin & released & bin_on_belt & bin_travelling
        self._deliver_sustain = self._advance_sustain("tray_to_bin_deliver", self._deliver_sustain, delivering)

        tray_lifted = torch.abs(tray_pos[:, 2] - self._tray_rest_z) > self._tray_z_tol
        tray_shoved = torch.abs(tray_pos[:, 0] - ox) > self._tray_x_tol
        tray_tipped = quaternion_to_matrix(tray_pose.q)[:, 2, 2] < math.cos(math.radians(self._tray_tilt_deg))
        tray_on_belt = torch.abs(tray_pos[:, 1] - oy) <= self._conveyor_half_length - self._tray_outer_half_y
        self._tray_disturbed = self._tray_disturbed | (
            (tray_lifted | tray_shoved | tray_tipped) & tray_on_belt & ~self._delivered
        )
        self._delivered = self._delivered | (in_bin_xy & in_bin_z & released)

        on_deck = torch.abs((part_pos[:, 2] - self._part_half_z) - deck_z) < self._deck_rest_tol
        between_carriers = self._outside_footprint(
            tray_pose, self._tray_outer_half_x, self._tray_outer_half_y, part_pos
        ) & self._outside_footprint(bin_pose, self._bin_outer_half_x, self._bin_outer_half_y, part_pos)
        self._deck_rested = self._deck_rested | (on_deck & between_carriers)

        success = (self._deliver_sustain >= self._deliver_sustain_steps) & ~self._tray_disturbed & ~self._deck_rested

        return {
            "success": success,
            "in_bin_xy": in_bin_xy,
            "in_bin_z": in_bin_z,
            "riding_with_bin": riding_with_bin,
            "released": released,
            "fingers_open": fingers_open,
            "hand_clear": hand_clear,
            "bin_on_belt": bin_on_belt,
            "bin_travelling": bin_travelling,
            "delivering": delivering,
            "deliver_sustain": self._deliver_sustain,
            "delivered": self._delivered,
            "tray_disturbed": self._tray_disturbed,
            "deck_rested": self._deck_rested,
            "tray_along_slip": torch.abs(self.tray.linear_velocity[:, 1] - belt_v),
            "part_to_bin_xy": torch.linalg.norm(local[:, :2], dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p
        bin_pos = self.bin.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - part_pos, dim=-1))
        carry_reward = 1 - torch.tanh(3 * torch.linalg.norm(part_pos[:, :2] - bin_pos[:, :2], dim=-1))
        in_bin = (info["in_bin_xy"] & info["in_bin_z"]).float()
        clean = (~info["tray_disturbed"] & ~info["deck_rested"]).float()
        return reach_reward + carry_reward + 2.0 * in_bin * clean

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
