from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed, tolerance_speed
from sweep_bench.tasks.conveyor_pin_release.assets import (
    build_pinned_carrier,
    build_retaining_pin,
    build_spring_flap,
)
from sweep_bench.tasks.conveyor_pin_release.config import RandomizationConfig
from sweep_bench.tasks.conveyor_pin_release.scripted import ScriptedPinReleaseMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorPinRelease-v0")
@register_env("ConveyorPinRelease-v0", max_episode_steps=300)
class ConveyorPinReleaseEnv(ScriptedPinReleaseMixin, ConveyorWorkspaceEnv):
    """Pull the retaining pin so a spring-loaded flap on a belt rider snaps open."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _carrier_half_x: float = 0.050
    _carrier_half_y: float = 0.048
    _floor_half_thickness: float = 0.003
    _wall_half_thickness: float = 0.002
    _carrier_density: float = 1800.0
    _carrier_patch_radius: float = 0.030

    _hinge_y: float = -0.016
    _hinge_z: float = 0.028
    _hinge_pin_radius: float = 0.004
    _hinge_clearance: float = 0.0006
    _journal_inner_x: float = 0.026
    _journal_outer_x: float = 0.034

    _bore_y: float = -0.040
    _bore_z: float = 0.019
    _keeper_x_lo: float = -0.014
    _keeper_x_hi: float = 0.000

    _plate_half_x: float = 0.020
    _plate_half_y: float = 0.028
    _plate_half_thickness: float = 0.002
    _plate_root_gap: float = 0.006
    _hinge_pin_half_length: float = 0.039
    _collar_radius: float = 0.006
    _collar_half_length: float = 0.0015
    _collar_x: float = 0.036
    _flap_density: float = 400.0
    _bearing_friction: float = 0.05

    _tail_x_lo: float = 0.004
    _tail_x_hi: float = 0.016
    _tail_y_near: float = -0.006
    _tail_y_far: float = -0.027
    _tail_half_thickness: float = 0.001
    _tail_underside: float = -0.0045

    _pin_radius: float = 0.004
    _pin_post_radius: float = 0.006
    _pin_post_half_height: float = 0.016
    _pin_grip_z: float = 0.022
    _pin_density: float = 3000.0
    _pin_grip_friction: float = 0.25
    _pin_tip_overhang: float = 0.002

    _flap_open_angle: float = math.radians(35.0)
    _pin_clear_margin: float = 0.008
    _pin_grip_tol: float = 0.003
    _pin_comove_speed: float = 0.04
    _pin_hold_z_above: float = 0.10
    _flap_hinge_capture: float = 0.010
    _carrier_lift_tol: float = 0.006
    _carrier_max_tilt_cos: float = math.cos(math.radians(10.0))
    _carrier_speed_tol_base: float = 0.06
    _carrier_speed_tol_per_speed: float = 0.20
    _carrier_min_speed_frac: float = 0.5
    _release_sustain_steps: int = 6

    _panda_finger_outboard_reach: float = 0.0152
    _keeper_finger_margin: float = 0.002

    _flap_spring_stiffness: float = 0.010
    _flap_spring_rest_angle: float = 1.0
    _flap_spring_damping: float = 0.0005
    _flap_spring_torque_cap: float = 0.02
    _flap_spring_ramp_steps: int = 5

    default_prompt: str = "pull the retaining pin out of the carrier on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        protrusion: float = 0.030,
        bore_clearance: float = 0.0014,
        **kwargs,
    ) -> None:
        min_protrusion = (
            0.5 * self._scripted_open_finger_gap + self._panda_finger_outboard_reach + self._keeper_finger_margin
        )
        if protrusion < min_protrusion:
            raise ValueError(
                f"protrusion {protrusion} is below the finger-vs-keeper floor {min_protrusion:.4f} "
                f"(open finger gap {self._scripted_open_finger_gap} + finger outboard reach "
                f"{self._panda_finger_outboard_reach} + margin {self._keeper_finger_margin}): "
                "the descending open fingertip would land on the keeper bracket"
            )
        if bore_clearance <= 0.0:
            raise ValueError(f"bore_clearance must be positive (got {bore_clearance})")
        self.protrusion: float = protrusion
        self.bore_clearance: float = bore_clearance
        super().__init__(*args, **kwargs)

    @property
    def _pin_grip_x(self) -> float:
        return self._keeper_x_lo - self.protrusion

    @property
    def _pin_shank_length(self) -> float:
        return self._tail_x_hi + self._pin_tip_overhang - self._pin_grip_x

    @property
    def _pin_clear_tip_x(self) -> float:
        return self._keeper_x_lo - self._pin_clear_margin

    @property
    def _bore_clear(self) -> float:
        return self._pin_radius + self.bore_clearance

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_pinned_carrier(
            self.scene,
            half_x=self._carrier_half_x,
            half_y=self._carrier_half_y,
            floor_half_thickness=self._floor_half_thickness,
            hinge_y=self._hinge_y,
            hinge_z=self._hinge_z,
            hinge_pin_radius=self._hinge_pin_radius,
            hinge_clearance=self._hinge_clearance,
            journal_inner_x=self._journal_inner_x,
            journal_outer_x=self._journal_outer_x,
            wall_half_thickness=self._wall_half_thickness,
            bore_y=self._bore_y,
            bore_z=self._bore_z,
            bore_clear=self._bore_clear,
            keeper_x_lo=self._keeper_x_lo,
            keeper_x_hi=self._keeper_x_hi,
            density=self._carrier_density,
            bearing_friction=self._bearing_friction,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.carrier = carrier_builder.build(name="pin_release_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        flap_builder = build_spring_flap(
            self.scene,
            plate_half_x=self._plate_half_x,
            plate_half_y=self._plate_half_y,
            plate_half_thickness=self._plate_half_thickness,
            plate_root_gap=self._plate_root_gap,
            tail_x_lo=self._tail_x_lo,
            tail_x_hi=self._tail_x_hi,
            tail_y_near=self._tail_y_near,
            tail_y_far=self._tail_y_far,
            tail_half_thickness=self._tail_half_thickness,
            tail_underside=self._tail_underside,
            hinge_pin_radius=self._hinge_pin_radius,
            hinge_pin_half_length=self._hinge_pin_half_length,
            collar_radius=self._collar_radius,
            collar_half_length=self._collar_half_length,
            collar_x=self._collar_x,
            density=self._flap_density,
            bearing_friction=self._bearing_friction,
        )
        flap_builder.initial_pose = sapien.Pose(p=[ox, oy + self._hinge_y, deck_z + self._hinge_z])
        self.flap = flap_builder.build(name="spring_flap")

        pin_builder = build_retaining_pin(
            self.scene,
            shank_radius=self._pin_radius,
            shank_length=self._pin_shank_length,
            shank_dz=self._bore_z - self._pin_grip_z,
            post_radius=self._pin_post_radius,
            post_half_height=self._pin_post_half_height,
            density=self._pin_density,
            grip_friction=self._pin_grip_friction,
            bore_friction=self._bearing_friction,
        )
        pin_builder.initial_pose = sapien.Pose(p=[ox + self._pin_grip_x, oy + self._bore_y, deck_z + self._pin_grip_z])
        self.pin = pin_builder.build(name="retaining_pin")

        tip = torch.zeros((self.num_envs, 3), device=self.device)
        tip[:, 0] = self._pin_shank_length
        tip[:, 2] = self._bore_z - self._pin_grip_z
        self._pin_tip_offset = Pose.create_from_pq(p=tip)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.flap, self.pin]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_q = self._sample_carrier_orientation(env_idx, b)
        carrier_pos = self._sample_carrier_position(env_idx, b)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        rot = quaternion_to_matrix(carrier_q)

        hinge_local = torch.zeros((b, 3), device=device)
        hinge_local[:, 1] = self._hinge_y
        hinge_local[:, 2] = self._hinge_z
        self.flap.set_pose(
            Pose.create_from_pq(carrier_pos + torch.bmm(rot, hinge_local.unsqueeze(-1)).squeeze(-1), carrier_q)
        )
        self.flap.set_linear_velocity(vel)
        self.flap.set_angular_velocity(torch.zeros((b, 3), device=device))

        pin_local = torch.zeros((b, 3), device=device)
        pin_local[:, 0] = self._pin_grip_x
        pin_local[:, 1] = self._bore_y
        pin_local[:, 2] = self._pin_grip_z
        self.pin.set_pose(
            Pose.create_from_pq(carrier_pos + torch.bmm(rot, pin_local.unsqueeze(-1)).squeeze(-1), carrier_q)
        )
        self.pin.set_linear_velocity(vel)
        self.pin.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._sample_flap_spring(env_idx, b)
        self._reset_release_trackers(env_idx)
        self._flap_spring_ready = True

    def _sample_carrier_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.carrier_yaw:
            yaw_np = self._episode_rng.uniform(*self.randomization.yaw_range, size=b).astype(np.float32)
            yaw = torch.as_tensor(yaw_np, device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z
        return pos

    def _sample_flap_spring(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_flap_spring_k"):
            self._flap_spring_k = torch.full(
                (self.num_envs,), self._flap_spring_stiffness, device=self.device, dtype=torch.float32
            )
        if self.randomization.flap_spring:
            k = self._episode_rng.uniform(*self.randomization.spring_range, size=b).astype(np.float32)
            self._flap_spring_k[env_idx] = torch.as_tensor(k, device=self.device)
        else:
            self._flap_spring_k[env_idx] = self._flap_spring_stiffness

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_flap_spring_torque()

    def _apply_flap_spring_torque(self) -> None:
        if not getattr(self, "_flap_spring_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        axis = quaternion_to_matrix(self.carrier.pose.q)[:, :, 0]
        theta = self._flap_angle()
        omega = ((self.flap.angular_velocity - self.carrier.angular_velocity) * axis).sum(dim=-1)
        tau = (self._flap_spring_k * (self._flap_spring_rest_angle - theta) - self._flap_spring_damping * omega).clamp(
            -self._flap_spring_torque_cap, self._flap_spring_torque_cap
        )
        ramp = ((self.elapsed_steps.to(tau.dtype) + 1.0) / self._flap_spring_ramp_steps).clamp(max=1.0)
        tau = tau * ramp
        torque = axis * tau.unsqueeze(-1)
        if gpu:
            torque_buf = self.scene.px.cuda_rigid_body_torque.torch()
            torque_buf[self.flap._body_data_index, :3] = torque
            self.scene.px.gpu_apply_rigid_dynamic_torque()
        else:
            self.flap._bodies[0].add_force_torque(np.zeros(3, dtype=np.float32), torque.squeeze(0).cpu().numpy())

    def _carrier_rotation(self) -> torch.Tensor:
        return quaternion_to_matrix(self.carrier.pose.q)

    def _flap_angle(self) -> torch.Tensor:
        rel = torch.bmm(self._carrier_rotation().transpose(1, 2), quaternion_to_matrix(self.flap.pose.q))
        return torch.atan2(rel[:, 2, 1], rel[:, 1, 1])

    def _carrier_local(self, world_points: torch.Tensor) -> torch.Tensor:
        rel = world_points - self.carrier.pose.p
        return torch.bmm(self._carrier_rotation().transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)

    def _pin_tip_local_x(self) -> torch.Tensor:
        return self._carrier_local((self.pin.pose * self._pin_tip_offset).p)[:, 0]

    def _flap_hinge_offset(self) -> torch.Tensor:
        anchor = torch.tensor([0.0, self._hinge_y, self._hinge_z], device=self.device)
        return torch.linalg.norm(self._carrier_local(self.flap.pose.p) - anchor, dim=-1)

    def _pin_tcp_offset(self) -> torch.Tensor:
        return self.pin.pose.p - cast("Panda", self.agent).tcp.pose.p

    def _pin_comove_rate(self) -> torch.Tensor:
        offset = self._pin_tcp_offset()
        rate = torch.linalg.norm(offset - self._pin_offset_prev, dim=-1) * self.control_freq
        advanced = self._eval_advanced("pin_comove").unsqueeze(-1)
        self._pin_offset_prev = torch.where(advanced, offset, self._pin_offset_prev)
        return rate

    def _pin_is_held(self) -> torch.Tensor:
        separation = self.gripper_aperture()
        pinched = torch.abs(separation - 2.0 * self._pin_post_radius) < self._pin_grip_tol
        return pinched & (self._pin_comove_rate() < self._pin_comove_speed)

    def _reset_release_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_release_sustain_counter"):
            self._release_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._carrier_ever_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._pin_ever_withdrawn = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._pin_offset_prev = torch.zeros((self.num_envs, 3), device=self.device)
        self._release_sustain_counter[env_idx] = 0
        self._carrier_ever_lifted[env_idx] = False
        self._pin_ever_withdrawn[env_idx] = False
        self._pin_offset_prev[env_idx] = self._pin_tcp_offset()[env_idx]

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        carrier_pos = self.carrier.pose.p

        angle = self._flap_angle()
        tip_x = self._pin_tip_local_x()
        hinge_offset = self._flap_hinge_offset()

        flap_open = angle >= self._flap_open_angle
        self._pin_ever_withdrawn = self._pin_ever_withdrawn | (tip_x <= self._pin_clear_tip_x)
        pin_withdrawn = self._pin_ever_withdrawn
        pin_high = self.pin.pose.p[:, 2] >= deck_z + self._pin_hold_z_above
        pin_held = self._pin_is_held() & pin_high
        flap_hinged = hinge_offset < self._flap_hinge_capture

        lifted_now = torch.abs(carrier_pos[:, 2] - deck_z) > self._carrier_lift_tol
        self._carrier_ever_lifted = self._carrier_ever_lifted | lifted_now
        never_lifted = ~self._carrier_ever_lifted

        on_belt = (
            (carrier_pos[:, 2] > deck_z - self._conveyor_deck_height * 0.5)
            & (torch.abs(carrier_pos[:, 0] - ox) < self._conveyor_half_width + self._carrier_half_x)
            & (torch.abs(carrier_pos[:, 1] - oy) < self._conveyor_half_length + self._carrier_half_y)
        )

        belt_v = self.belt.velocity
        belt_speed = torch.abs(belt_v)
        carrier_vy = self.carrier.linear_velocity[:, 1]
        speed_tol = self._carrier_speed_tol_base + self._carrier_speed_tol_per_speed * tolerance_speed(belt_v)
        tracking = torch.abs(carrier_vy - belt_v) <= speed_tol
        min_speed = carried_speed(belt_speed, self._carrier_min_speed_frac)
        not_jammed = (torch.abs(carrier_vy) >= min_speed) & (carrier_vy * belt_v >= 0.0)
        upright = self._carrier_rotation()[:, 2, 2] > self._carrier_max_tilt_cos
        carrier_riding = never_lifted & on_belt & tracking & not_jammed & upright

        holding = flap_open & pin_withdrawn & pin_held & flap_hinged & carrier_riding
        self._release_sustain_counter = self._advance_sustain(
            "pin_release_sustain", self._release_sustain_counter, holding
        )
        success = self._release_sustain_counter >= self._release_sustain_steps

        return {
            "success": success,
            "flap_angle": angle,
            "flap_open": flap_open,
            "pin_tip_x": tip_x,
            "pin_withdrawn": pin_withdrawn,
            "pin_held": pin_held,
            "flap_hinge_offset": hinge_offset,
            "flap_hinged": flap_hinged,
            "carrier_never_lifted": never_lifted,
            "carrier_on_belt": on_belt,
            "carrier_tracking": tracking,
            "carrier_not_jammed": not_jammed,
            "carrier_riding": carrier_riding,
            "release_sustain": self._release_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_post = torch.linalg.norm(tcp_pos - self.pin.pose.p, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_post)

        seated_tip_x = self._tail_x_hi + self._pin_tip_overhang
        travel = seated_tip_x - self._pin_clear_tip_x
        extract_reward = ((seated_tip_x - info["pin_tip_x"]) / travel).clamp(0.0, 1.0)

        release_reward = (info["flap_angle"] / self._flap_open_angle).clamp(0.0, 1.0)

        return reach_reward + 2.0 * extract_reward * info["pin_held"].float() + 2.0 * release_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
