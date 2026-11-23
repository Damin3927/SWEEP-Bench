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
from sweep_bench.tasks.conveyor_ramp_roll.assets import (
    build_ramp_carrier,
    ramp_surface_z,
    seat_height,
)
from sweep_bench.tasks.conveyor_ramp_roll.config import RandomizationConfig
from sweep_bench.tasks.conveyor_ramp_roll.scripted import ScriptedRampRollMixin
from sweep_bench.tasks.conveyor_ramp_roll.variants import BALL_SPECS, RAMP_ROLL_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorRampRoll-v0")
@register_env("ConveyorRampRoll-v0", max_episode_steps=300)
class ConveyorRampRollEnv(ScriptedRampRollMixin, ConveyorWorkspaceEnv):
    """Set the held ball down at the top of a belt-riding ramp and let it run into the catch well."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(RAMP_ROLL_OBJECT_IDS, extra=BALL_SPECS)

    conveyor_speed: float = 0.10

    _ramp_angle: float = math.radians(18.0)
    _ramp_run_above: float = 0.030
    _ramp_run_below: float = 0.110
    _pocket_length: float = 0.055
    _pocket_depth: float = 0.030
    _pocket_rim_above_ramp: float = 0.006
    _channel_side_gap: float = 0.006
    _channel_wall_height: float = 0.010
    _wall_thickness: float = 0.006
    _base_thickness: float = 0.010
    _ramp_thickness: float = 0.022

    _carrier_patch_radius: float = 0.050
    _ball_patch_radius: float = 0.02

    _caught_floor_tol: float = 0.004
    _caught_rel_speed_frac: float = 0.25
    _caught_rel_speed_floor: float = 0.05
    _caught_rel_speed_ceiling: float = 0.125
    _transit_min_travel: float = 0.12
    _escort_min_distance: float = 0.06
    _escort_slack: float = 0.010
    _channel_hop_margin: float = 0.025
    _release_aperture_margin: float = 0.006
    _carrier_tilt_cos: float = math.cos(math.radians(10.0))
    _carrier_x_tol: float = 0.05
    _carrier_comove_tol_frac: float = 0.25
    _carrier_comove_tol_floor: float = 0.040
    _carrier_comove_tol_ceiling: float = 0.125
    _catch_sustain_steps: int = 8

    default_prompt: str = "release the ball at the top of the ramp so it rolls into the catch pocket"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _ramp_top_local_z(self) -> float:
        return ramp_surface_z(-self._ramp_run_above, self._ramp_angle)

    @property
    def _ramp_bottom_local_z(self) -> float:
        return ramp_surface_z(self._ramp_run_below, self._ramp_angle)

    @property
    def _pocket_floor_local_z(self) -> float:
        return self._ramp_bottom_local_z - self._pocket_depth

    @property
    def _pocket_y_lo(self) -> float:
        return self._ramp_run_below

    @property
    def _pocket_y_hi(self) -> float:
        return self._ramp_run_below + self._pocket_length

    @property
    def _release_max_local_y(self) -> float:
        return 0.5 * self._ramp_run_below

    @property
    def _carrier_origin_above_deck(self) -> float:
        return -self._pocket_floor_local_z + self._base_thickness

    @property
    def _carrier_span_back(self) -> float:
        return self._ramp_run_above + 2.0 * self._wall_thickness

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._ball_names: list[str] = [str(name) for name in names]
        self._ball_display: list[str] = [display_name(spec_by_id[name]) for name in self._ball_names]
        specs = [spec_by_id[name] for name in self._ball_names]
        radii = [float(spec.bbox[0]) for spec in specs]

        balls: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.50])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"ball-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            balls.append(obj)
        self.ball = Actor.merge(balls, name="ball")
        self.add_to_state_dict_registry(self.ball)
        self.belt.register_rider(
            self.ball,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._ball_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        carriers: list[Actor] = []
        for i, radius in enumerate(radii):
            builder = build_ramp_carrier(
                self.scene,
                ball_radius=radius,
                angle=self._ramp_angle,
                run_above=self._ramp_run_above,
                run_below=self._ramp_run_below,
                pocket_length=self._pocket_length,
                pocket_depth=self._pocket_depth,
                pocket_rim_above_ramp=self._pocket_rim_above_ramp,
                channel_side_gap=self._channel_side_gap,
                channel_wall_height=self._channel_wall_height,
                wall_thickness=self._wall_thickness,
                base_thickness=self._base_thickness,
                ramp_thickness=self._ramp_thickness,
                release_gap=self._scripted_release_gap,
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._carrier_origin_above_deck])
            builder.set_scene_idxs([i])
            carrier = builder.build(name=f"ramp_carrier-{i}")
            self.remove_from_state_dict_registry(carrier)
            carriers.append(carrier)
        self.carrier = Actor.merge(carriers, name="ramp_carrier")
        self.add_to_state_dict_registry(self.carrier)
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        device = self.device
        self._ball_radius = torch.as_tensor(radii, dtype=torch.float32, device=device)
        self._ball_seat_local_z = torch.as_tensor(
            [seat_height(r, self._ramp_angle) for r in radii], dtype=torch.float32, device=device
        )
        self._channel_half_width = self._ball_radius + self._channel_side_gap

    @property
    def episode_objects(self) -> list[str]:
        return list(self._ball_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"release the {name} at the top of the ramp so it rolls into the catch pocket"
            for name in self._ball_display
        ]

    @property
    def _primary_rider(self) -> Actor:
        return self.carrier

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        radius = self._ball_radius[env_idx].cpu().numpy()
        return 2.0 * (radius - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_pos = self._sample_carrier_position(env_idx, b)
        yaw = torch.where(
            self.belt.direction_sign[env_idx] > 0.0,
            torch.zeros(b, device=device),
            torch.full((b,), math.pi, device=device),
        )
        if self.randomization.carrier_yaw:
            lo, hi = self.randomization.carrier_yaw_range
            jitter = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            yaw = yaw + torch.as_tensor(jitter, device=device)
        carrier_q = torch.zeros((b, 4), device=device)
        carrier_q[:, 0] = torch.cos(0.5 * yaw)
        carrier_q[:, 3] = torch.sin(0.5 * yaw)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        carrier_vel = torch.zeros((b, 3), device=device)
        carrier_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(carrier_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        ball_q = torch.zeros((b, 4), device=device)
        ball_q[:, 0] = 1.0
        self.ball.set_pose(Pose.create_from_pq(tcp_pos, ball_q))
        self.ball.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.ball.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_ramp_trackers(env_idx)

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_span_back - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._carrier_origin_above_deck
        return pos

    def _reset_ramp_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_ball_released"):
            b, device = self.num_envs, self.device
            self._ball_released = torch.zeros(b, dtype=torch.bool, device=device)
            self._release_local = torch.zeros((b, 3), device=device)
            self._release_station_ok = torch.zeros(b, dtype=torch.bool, device=device)
            self._escort_peak_travel = torch.zeros(b, device=device)
            self._no_escort = torch.ones(b, dtype=torch.bool, device=device)
            self._ball_at_rest = torch.zeros(b, dtype=torch.bool, device=device)
            self._ball_in_channel = torch.ones(b, dtype=torch.bool, device=device)
            self._catch_sustain_counter = torch.zeros(b, dtype=torch.long, device=device)
        self._ball_released[env_idx] = False
        self._release_local[env_idx] = 0.0
        self._release_station_ok[env_idx] = False
        self._escort_peak_travel[env_idx] = 0.0
        self._no_escort[env_idx] = True
        self._ball_at_rest[env_idx] = False
        self._ball_in_channel[env_idx] = True
        self._catch_sustain_counter[env_idx] = 0

    def evaluate(self):
        agent = cast("Panda", self.agent)
        carrier_pose = self.carrier.pose
        carrier_pos = carrier_pose.p
        ball_pos = self.ball.pose.p
        tcp_pos = agent.tcp.pose.p

        local = (carrier_pose.inv() * Pose.create_from_pq(ball_pos)).p

        finger_width = self.gripper_aperture()
        opened = finger_width > 2.0 * self._ball_radius + self._release_aperture_margin
        newly_released = opened & ~self._ball_released
        self._release_local = torch.where(newly_released.unsqueeze(-1), local, self._release_local)
        station_ok = (local[:, 1] <= self._release_max_local_y) & (local[:, 1] >= -self._ramp_run_above)
        self._release_station_ok = torch.where(newly_released, station_ok, self._release_station_ok)
        self._ball_released = self._ball_released | opened

        travel = torch.linalg.norm(local - self._release_local, dim=-1)
        travelled = self._ball_released & (travel >= self._transit_min_travel)

        rel_speed = torch.linalg.norm(self.ball.linear_velocity - self.carrier.linear_velocity, dim=-1)
        caught_tol = torch.clamp(
            self._caught_rel_speed_frac * tolerance_speed(self.belt.velocity),
            min=self._caught_rel_speed_floor,
            max=self._caught_rel_speed_ceiling,
        )
        in_pocket = (
            (local[:, 1] >= self._pocket_y_lo)
            & (local[:, 1] <= self._pocket_y_hi)
            & (torch.abs(local[:, 0]) <= self._channel_half_width)
            & (torch.abs(local[:, 2] - self._ball_radius - self._pocket_floor_local_z) <= self._caught_floor_tol)
        )
        caught = in_pocket & (rel_speed <= caught_tol)

        self._escort_peak_travel = torch.where(
            self._ball_released,
            torch.maximum(self._escort_peak_travel, travel),
            self._escort_peak_travel,
        )
        escort_required = torch.clamp(
            self._escort_peak_travel - self._escort_slack, min=0.0, max=self._escort_min_distance
        )
        tcp_ball_dist = torch.linalg.norm(tcp_pos - ball_pos, dim=-1)
        escorting = self._ball_released & ~self._ball_at_rest & (tcp_ball_dist < escort_required)
        self._no_escort = self._no_escort & ~escorting
        self._ball_at_rest = self._ball_at_rest | (caught & (self._escort_peak_travel >= self._transit_min_travel))

        ceiling = self._ball_seat_local_z + self._scripted_release_gap + self._channel_hop_margin
        on_the_run = (torch.abs(local[:, 0]) <= self._channel_half_width) & (local[:, 2] <= ceiling)
        self._ball_in_channel = self._ball_in_channel & (~self._ball_released | on_the_run)

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        up_local = torch.zeros((self.num_envs, 3), device=self.device)
        up_local[:, 2] = 1.0
        tilt_cos = ((carrier_pose * Pose.create_from_pq(up_local)).p - carrier_pos)[:, 2]
        comove_tol = torch.clamp(
            self._carrier_comove_tol_frac * tolerance_speed(self.belt.velocity),
            min=self._carrier_comove_tol_floor,
            max=self._carrier_comove_tol_ceiling,
        )
        carrier_riding = (
            (carrier_pos[:, 2] > deck_z + 0.5 * self._carrier_origin_above_deck)
            & (torch.abs(carrier_pos[:, 0] - ox) <= self._carrier_x_tol)
            & (torch.abs(carrier_pos[:, 1] - oy) <= self._conveyor_half_length)
            & (tilt_cos >= self._carrier_tilt_cos)
            & (torch.abs(self.carrier.linear_velocity[:, 1] - self.belt.velocity) <= comove_tol)
        )

        landed = (
            caught & travelled & self._release_station_ok & self._no_escort & self._ball_in_channel & carrier_riding
        )

        self._catch_sustain_counter = self._advance_sustain("catch_sustain", self._catch_sustain_counter, landed)
        success = self._catch_sustain_counter >= self._catch_sustain_steps

        return {
            "success": success,
            "caught": caught,
            "travelled": travelled,
            "release_station_ok": self._release_station_ok,
            "no_escort": self._no_escort,
            "ball_in_channel": self._ball_in_channel,
            "carrier_riding": carrier_riding,
            "ball_released": self._ball_released,
            "ball_travel": travel,
            "ball_run_y": local[:, 1],
            "ball_rel_speed": rel_speed,
            "tcp_ball_dist": tcp_ball_dist,
            "escort_required": escort_required,
            "release_height_above_pocket": self._release_local[:, 2] - self._pocket_floor_local_z,
            "catch_sustain": self._catch_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        ball_pos = self.ball.pose.p
        carrier_pos = self.carrier.pose.p

        xy_dist = torch.linalg.norm(ball_pos[:, :2] - carrier_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        run_reward = (info["ball_travel"] / self._transit_min_travel).clamp(0.0, 1.0)

        return carry_reward + 2.0 * run_reward * info["ball_released"].float() + 2.0 * info["caught"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
