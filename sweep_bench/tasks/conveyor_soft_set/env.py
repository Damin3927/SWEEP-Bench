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
from sweep_bench.tasks.conveyor_soft_set.assets import (
    build_carrier_well,
    build_soft_platform,
)
from sweep_bench.tasks.conveyor_soft_set.config import RandomizationConfig
from sweep_bench.tasks.conveyor_soft_set.scripted import ScriptedSoftSetMixin
from sweep_bench.tasks.conveyor_soft_set.variants import SOFT_SET_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec

_G = 9.81

_SPRING_DAMPING_RATIO = 0.25

_FIT_EPS = 1e-6


@bench_task("ConveyorSoftSet-v0")
@register_env("ConveyorSoftSet-v0", max_episode_steps=300)
class ConveyorSoftSetEnv(ScriptedSoftSetMixin, ConveyorWorkspaceEnv):
    """Set a held part down on a sprung platform gently enough that it barely moves."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(SOFT_SET_OBJECT_IDS)

    _carrier_half_width: float = 0.057
    _bore_half_width: float = 0.0486
    _carrier_wall_height: float = 0.036
    _carrier_floor_half_thickness: float = 0.004

    _platform_half_width: float = 0.048
    _platform_half_thickness: float = 0.021
    _platform_rest_gap: float = 0.016

    _spring_stiffness: float = 400.0
    _spring_force_max: float = 20.0

    _carrier_patch_radius: float = 0.035
    _part_patch_radius: float = 0.02

    _part_mass_min: float = 0.060
    _part_mass_max: float = 0.100
    _part_min_half_height: float = 0.018
    _part_max_half_height: float = 0.030
    _max_finger_aperture: float = 0.08

    _seated_support_margin: float = 0.002
    _seated_min_tol: float = 0.016
    _seated_gap_tol: float = 0.003
    _seated_rel_speed: float = 0.05
    _released_finger_margin: float = 0.012
    _released_distance: float = 0.08
    _gentle_arm_radius: float = 0.05
    _gentle_load_factor: float = 2.8
    _carrier_tilt_cos: float = math.cos(math.radians(10.0))
    _carrier_x_tol: float = 0.05
    _carrier_y_limit: float = 0.85
    _carrier_speed_tol_floor: float = 0.06
    _carrier_speed_tol_frac: float = 0.25
    _seat_sustain_steps: int = 8

    default_prompt: str = "set the held object down gently on the pad on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _carrier_origin_above_deck(self) -> float:
        return 2.0 * self._carrier_floor_half_thickness

    @property
    def _platform_anchor_local_z(self) -> float:
        return self._platform_rest_gap + self._platform_half_thickness

    @property
    def _platform_top_above_carrier(self) -> float:
        return self._platform_anchor_local_z + self._platform_half_thickness

    def _check_part_fits(self, spec: ObjectSpec) -> None:
        footprint = max(spec.bbox[0], spec.bbox[1])
        grasp_width = spec.grasp_width if spec.grasp_width is not None else 2.0 * footprint
        needed = grasp_width + self._released_finger_margin
        if needed > self._max_finger_aperture + _FIT_EPS:
            raise ValueError(
                f"part {spec.id!r} needs a {needed} m gripper aperture to read as RELEASED, over the "
                f"{self._max_finger_aperture} m the hand can open to; the success predicate would be "
                "unreachable for it (see variants.py)"
            )
        if spec.mass is None or not (self._part_mass_min - _FIT_EPS <= spec.mass <= self._part_mass_max + _FIT_EPS):
            raise ValueError(
                f"part {spec.id!r} has mass {spec.mass} kg, outside the "
                f"[{self._part_mass_min}, {self._part_mass_max}] kg band the gentleness calibration is "
                f"stated for (see variants.py)"
            )
        radius = self._platform_half_width - footprint - self._seated_support_margin
        if radius < self._seated_min_tol - _FIT_EPS:
            raise ValueError(
                f"part {spec.id!r} has horizontal half-extent {footprint} m, which leaves a seating radius of "
                f"only {radius} m against the {self._seated_min_tol} m floor, so it could not be seated at the "
                f"top of the belt-speed band"
            )
        if spec.bbox[2] < self._part_min_half_height - _FIT_EPS:
            raise ValueError(
                f"part {spec.id!r} is only {spec.bbox[2]} m half-tall; below {self._part_min_half_height} m "
                f"the opening fingers rake the pad at the release height"
            )
        if spec.bbox[2] > self._part_max_half_height + _FIT_EPS:
            raise ValueError(
                f"part {spec.id!r} is {spec.bbox[2]} m half-tall; above {self._part_max_half_height} m a "
                f"seated part sits outside the {self._gentle_arm_radius} m arming radius and the gentleness "
                f"term would never arm"
            )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_carrier_well(
            self.scene,
            bore_half_width=self._bore_half_width,
            outer_half_width=self._carrier_half_width,
            wall_height=self._carrier_wall_height,
            floor_half_thickness=self._carrier_floor_half_thickness,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._carrier_origin_above_deck])
        self.carrier = carrier_builder.build(name="soft_set_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        platform_builder = build_soft_platform(
            self.scene,
            plate_half_width=self._platform_half_width,
            plate_half_thickness=self._platform_half_thickness,
        )
        platform_builder.initial_pose = sapien.Pose(
            p=[ox, oy, deck_z + self._carrier_origin_above_deck + self._platform_anchor_local_z]
        )
        self.pad = platform_builder.build(name="soft_set_platform")

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        specs = [spec_by_id[name] for name in self._part_names]
        for spec in specs:
            self._check_part_fits(spec)

        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.50])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"part-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        self.part = Actor.merge(objs, name="part")
        self.add_to_state_dict_registry(self.part)
        self.belt.register_rider(
            self.part,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._part_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self._part_half_z = torch.as_tensor([s.bbox[2] for s in specs], dtype=torch.float32, device=self.device)
        self._part_footprint_half = torch.as_tensor(
            [max(s.bbox[0], s.bbox[1]) for s in specs], dtype=torch.float32, device=self.device
        )
        self._part_grasp_width = torch.as_tensor(
            [s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1]) for s in specs],
            dtype=torch.float32,
            device=self.device,
        )
        self._part_mass = torch.as_tensor([s.mass for s in specs], dtype=torch.float32, device=self.device)

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"set the {name} down gently on the pad on the conveyor belt" for name in self._part_display]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = 0.5 * self._part_grasp_width[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_q = self._sample_carrier_orientation(env_idx, b)

        carrier_pos = self._sample_carrier_position(env_idx, b)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        carrier_vel = torch.zeros((b, 3), device=device)
        carrier_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(carrier_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        platform_pos = carrier_pos.clone()
        platform_pos[:, 2] = carrier_pos[:, 2] + self._platform_anchor_local_z
        self.pad.set_pose(Pose.create_from_pq(platform_pos, carrier_q))
        self.pad.set_linear_velocity(carrier_vel)
        self.pad.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_spring(env_idx, b)

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

        self._reset_seat_trackers(env_idx)
        self._spring_ready = True

    def _sample_carrier_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.carrier_yaw:
            lo, hi = self.randomization.carrier_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_half_width - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.carrier_lateral_offset:
            lo, hi = self.randomization.carrier_x_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._carrier_origin_above_deck
        return pos

    def _reset_spring(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        if not hasattr(self, "_spring_preload"):
            zeros = torch.zeros(self.num_envs, device=device)
            self._spring_preload = zeros.clone()
            self._spring_damping = zeros.clone()

        part_mass = self._part_mass[env_idx]
        pad_mass = self.pad.mass.to(device)[env_idx]
        self._spring_preload[env_idx] = pad_mass * _G
        self._spring_damping[env_idx] = (
            2.0 * _SPRING_DAMPING_RATIO * torch.sqrt(self._spring_stiffness * (pad_mass + part_mass))
        )

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_platform_spring_force()
        self._update_gentle_peak()

    def _apply_platform_spring_force(self) -> None:
        if not getattr(self, "_spring_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        depression = self._platform_depression()
        v_rel_z = self.pad.linear_velocity[:, 2] - self.carrier.linear_velocity[:, 2]
        fz = (self._spring_stiffness * depression + self._spring_preload - self._spring_damping * v_rel_z).clamp(
            0.0, self._spring_force_max
        )
        force = torch.zeros((self.num_envs, 3), device=device)
        force[:, 2] = fz
        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            force_buf[self.pad._body_data_index, :3] = force
            self.scene.px.gpu_apply_rigid_dynamic_force()
        else:
            self.pad._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))

    def _platform_depression(self) -> torch.Tensor:
        rel_z = self.pad.pose.p[:, 2] - self.carrier.pose.p[:, 2]
        return self._platform_anchor_local_z - rel_z

    def _platform_surface_center(self) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 2] = self._platform_half_thickness
        return (self.pad.pose * Pose.create_from_pq(local)).p

    @property
    def _seated_xy_tol(self) -> torch.Tensor:
        return (self._platform_half_width - self._part_footprint_half - self._seated_support_margin).clamp(min=0.0)

    @property
    def _gentle_peak_limit(self) -> torch.Tensor:
        return self._gentle_load_factor * self._part_mass * _G / self._spring_stiffness

    def _update_gentle_peak(self) -> None:
        if not hasattr(self, "_gentle_peak"):
            return
        dist = torch.linalg.norm(self.part.pose.p - self._platform_surface_center(), dim=-1)
        self._gentle_armed = self._gentle_armed | (dist <= self._gentle_arm_radius)
        depression = self._platform_depression().clamp(min=0.0)
        self._gentle_peak = torch.where(
            self._gentle_armed, torch.maximum(self._gentle_peak, depression), self._gentle_peak
        )

    def _reset_seat_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_seat_sustain_counter"):
            self._seat_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._gentle_peak = torch.zeros(self.num_envs, device=self.device)
            self._gentle_armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._seat_sustain_counter[env_idx] = 0
        self._gentle_peak[env_idx] = 0.0
        self._gentle_armed[env_idx] = False

    def evaluate(self):
        agent = cast("Panda", self.agent)
        carrier_pose = self.carrier.pose
        carrier_pos = carrier_pose.p
        platform_pose = self.pad.pose
        part_pos = self.part.pose.p
        tcp_pos = agent.tcp.pose.p

        self._update_gentle_peak()

        local_carrier = (carrier_pose.inv() * Pose.create_from_pq(part_pos)).p
        seated_xy = torch.linalg.norm(local_carrier[:, :2], dim=-1)
        local_pad = (platform_pose.inv() * Pose.create_from_pq(part_pos)).p
        bottom_gap = torch.abs((local_pad[:, 2] - self._part_half_z) - self._platform_half_thickness)
        rel_speed = torch.linalg.norm(self.part.linear_velocity - self.carrier.linear_velocity, dim=-1)
        seated = (
            (seated_xy <= self._seated_xy_tol)
            & (bottom_gap <= self._seated_gap_tol)
            & (rel_speed < self._seated_rel_speed)
        )

        finger_width = self.gripper_aperture()
        released = (finger_width > self._part_grasp_width + self._released_finger_margin) & (
            torch.linalg.norm(tcp_pos - part_pos, dim=-1) >= self._released_distance
        )

        peak = self._gentle_peak
        gentle = peak <= self._gentle_peak_limit

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        up_local = torch.zeros((self.num_envs, 3), device=self.device)
        up_local[:, 2] = 1.0
        tilt_cos = ((carrier_pose * Pose.create_from_pq(up_local)).p - carrier_pos)[:, 2]
        speed_tol = torch.clamp(
            self._carrier_speed_tol_frac * tolerance_speed(self.belt.velocity), min=self._carrier_speed_tol_floor
        )
        carrier_riding = (
            (carrier_pos[:, 2] > deck_z)
            & (torch.abs(carrier_pos[:, 0] - ox) <= self._carrier_x_tol)
            & (torch.abs(carrier_pos[:, 1] - oy) < self._carrier_y_limit)
            & (tilt_cos >= self._carrier_tilt_cos)
            & (torch.abs(self.carrier.linear_velocity[:, 1] - self.belt.velocity) <= speed_tol)
        )

        set_down = seated & released & gentle & carrier_riding

        self._seat_sustain_counter = self._advance_sustain("seat_sustain", self._seat_sustain_counter, set_down)
        success = self._seat_sustain_counter >= self._seat_sustain_steps

        return {
            "success": success,
            "seated": seated,
            "released": released,
            "gentle": gentle,
            "carrier_riding": carrier_riding,
            "peak_depression": peak,
            "peak_load_factor": peak * self._spring_stiffness / (self._part_mass * _G),
            "gentle_armed": self._gentle_armed,
            "depression": self._platform_depression(),
            "seat_sustain": self._seat_sustain_counter,
            "part_to_carrier_xy": seated_xy,
        }

    def compute_dense_reward(self, obs, action, info):
        part_pos = self.part.pose.p
        surface = self._platform_surface_center()

        xy_dist = torch.linalg.norm(part_pos[:, :2] - surface[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        gentle_reward = (1.0 - info["peak_depression"] / self._gentle_peak_limit).clamp(0.0, 1.0)

        return carry_reward + 2.0 * gentle_reward * info["seated"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
