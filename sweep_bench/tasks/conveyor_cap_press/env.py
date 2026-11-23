from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_cap_press.assets import build_bead_jar, build_press_cap
from sweep_bench.tasks.conveyor_cap_press.config import RandomizationConfig
from sweep_bench.tasks.conveyor_cap_press.scripted import ScriptedCapPressMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_CAP_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.20, 0.35, 0.75, 1.0),
    (0.85, 0.25, 0.25, 1.0),
    (0.25, 0.60, 0.30, 1.0),
    (0.95, 0.75, 0.20, 1.0),
    (0.55, 0.30, 0.70, 1.0),
    (0.30, 0.30, 0.32, 1.0),
)


@bench_task("ConveyorCapPress-v0")
@register_env("ConveyorCapPress-v0", max_episode_steps=300)
class ConveyorCapPressEnv(ScriptedCapPressMixin, ConveyorWorkspaceEnv):
    """Press-fit a held cap onto a jar riding the conveyor belt, past its bead."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _jar_body_radius: float = 0.045
    _jar_body_half_height: float = 0.025
    _jar_neck_radius: float = 0.016
    _jar_bead_radius: float = 0.020
    _jar_bead_half_height: float = 0.0030
    _jar_bead_below_neck_top: float = 0.007
    _jar_density: float = 2500.0

    _cap_outer_radius: float = 0.0285
    _cap_skirt_height: float = 0.016
    _cap_lead_in_height: float = 0.010
    _cap_crown_half_thickness: float = 0.003
    _cap_knob_radius: float = 0.012
    _cap_knob_half_height: float = 0.009
    _cap_density: float = 1000.0

    _bead_depth_lo: float = 0.004
    _bead_depth_hi: float = 0.010
    _bead_stiffness: float = 800.0
    _bead_damping: float = 9.0
    _bead_creep_speed: float = 0.15
    _bead_clear_speed: float = 0.25
    _bead_force_max: float = 40.0
    _bead_engage_radius: float = 0.0185
    _bead_interference_range: tuple[float, float] = (1.0, 1.4)

    _jar_spawn_max_dist: float = 0.40

    _jar_patch_radius: float = 0.030

    _seat_success_depth: float = 0.010
    _cap_lateral_threshold: float = 0.005
    _cap_axis_cos_threshold: float = 0.985
    _jar_upright_cos: float = 0.95
    _released_distance_threshold: float = 0.08
    _released_aperture: float = 0.028
    _comoving_rel_speed_threshold: float = 0.3
    _seat_sustain_steps: int = 6

    default_prompt: str = "press the cap onto the jar on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        neck_height: float = 0.036,
        bore_clearance: float = 0.0025,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        if bore_clearance <= 0.0:
            raise ValueError(f"bore_clearance must be positive (got {bore_clearance})")
        min_neck = self._cap_skirt_height + self._cap_lead_in_height + 0.005
        if neck_height < min_neck:
            raise ValueError(
                f"neck_height={neck_height} is too short: the seated cap's bevel would reach the "
                f"shoulder (needs >= {min_neck} m for this cap)"
            )
        self.neck_height: float = neck_height
        self.bore_clearance: float = bore_clearance
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if (num_envs == 1 and randomization.cap_color) else 0
        super().__init__(
            *args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, randomization=randomization, **kwargs
        )

    @property
    def _cap_bore_radius(self) -> float:
        return self._jar_neck_radius + self.bore_clearance

    @property
    def _jar_origin_above_deck(self) -> float:
        return self._jar_body_half_height

    @property
    def _neck_top_local_z(self) -> float:
        return self._jar_body_half_height + self.neck_height

    @property
    def _cap_grip_to_throat(self) -> float:
        return self._cap_knob_half_height + 2.0 * self._cap_crown_half_thickness + self._cap_skirt_height

    @property
    def _cap_seat_depth(self) -> float:
        return self._cap_skirt_height

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        jar_builder = build_bead_jar(
            self.scene,
            body_radius=self._jar_body_radius,
            body_half_height=self._jar_body_half_height,
            neck_radius=self._jar_neck_radius,
            neck_height=self.neck_height,
            bead_radius=self._jar_bead_radius,
            bead_half_height=self._jar_bead_half_height,
            bead_below_neck_top=self._jar_bead_below_neck_top,
            density=self._jar_density,
        )
        jar_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._jar_origin_above_deck])
        self.jar = jar_builder.build(name="cap_press_jar")
        self.belt.register_rider(self.jar, patch_radius=self._jar_patch_radius)

        if self.randomization.cap_color:
            color_idx = np.asarray(self._batched_episode_rng.choice(len(_CAP_COLORS))).reshape(self.num_envs)
        else:
            color_idx = np.zeros(self.num_envs, dtype=int)
        caps: list[Actor] = []
        for i in range(self.num_envs):
            cap_builder = build_press_cap(
                self.scene,
                bore_radius=self._cap_bore_radius,
                outer_radius=self._cap_outer_radius,
                skirt_height=self._cap_skirt_height,
                lead_in_height=self._cap_lead_in_height,
                mouth_radius=self._cap_outer_radius,
                crown_half_thickness=self._cap_crown_half_thickness,
                knob_radius=self._cap_knob_radius,
                knob_half_height=self._cap_knob_half_height,
                density=self._cap_density,
                cap_color=_CAP_COLORS[int(color_idx[i])],
            )
            cap_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
            cap_builder.set_scene_idxs([i])
            obj = cap_builder.build(name=f"press_cap-{i}")
            self.remove_from_state_dict_registry(obj)
            caps.append(obj)
        self.cap = Actor.merge(caps, name="press_cap")
        self.add_to_state_dict_registry(self.cap)

        throat = torch.zeros((self.num_envs, 3), device=self.device)
        throat[:, 2] = -self._cap_grip_to_throat
        self._cap_throat_offset = Pose.create_from_pq(p=throat)
        neck_top = torch.zeros((self.num_envs, 3), device=self.device)
        neck_top[:, 2] = self._neck_top_local_z
        self._neck_top_offset = Pose.create_from_pq(p=neck_top)
        self._cap_mass = self.cap.mass.to(self.device)

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._cap_knob_radius - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        jar_pos = self._sample_jar_position(env_idx, b)
        jar_q = torch.zeros((b, 4), device=device)
        jar_q[:, 0] = 1.0
        self.jar.set_pose(Pose.create_from_pq(jar_pos, jar_q))
        jar_vel = torch.zeros((b, 3), device=device)
        jar_vel[:, 1] = self.belt.velocity[env_idx]
        self.jar.set_linear_velocity(jar_vel)
        self.jar.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        cap_q = torch.zeros((b, 4), device=device)
        cap_q[:, 0] = 1.0
        self.cap.set_pose(Pose.create_from_pq(tcp_pos, cap_q))
        self.cap.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.cap.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_press_trackers(env_idx, b)
        self._bead_ready = True

    def _sample_jar_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._jar_body_radius - 0.04,
            self._jar_spawn_max_dist,
        )
        jar_pos = torch.zeros((b, 3), device=device)
        jar_pos[:, 0] = ox
        if self.randomization.jar_lateral:
            lo, hi = self.randomization.jar_lateral_range
            offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            jar_pos[:, 0] = ox + torch.as_tensor(offsets, device=device)
        jar_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        jar_pos[:, 2] = deck_z + self._jar_origin_above_deck
        return jar_pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_bead_force()

    def _apply_bead_force(self) -> None:
        if not getattr(self, "_bead_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()

        depth, lateral = self._bead_engagement()
        v_rel = self.cap.linear_velocity[:, 2] - self.jar.linear_velocity[:, 2]

        advance = (depth - self._bead_prev_depth) * self.sim_freq
        walked = self._bead_prev_engaged & (self._bead_prev_depth < self._bead_depth_hi)
        self._bead_cleared = self._bead_cleared | (
            walked & (depth >= self._bead_depth_hi) & (advance <= self._bead_clear_speed)
        )
        engaged = (~self._bead_cleared) & (lateral < self._bead_engage_radius) & (depth > self._bead_depth_lo)
        self._bead_prev_depth = depth
        self._bead_prev_engaged = engaged

        excess = (-v_rel - self._bead_creep_speed).clamp(min=0.0)
        resist = (
            self._bead_stiffness * self._bead_force_scale * (depth - self._bead_depth_lo)
            - self._bead_damping * v_rel
            + self._cap_mass * excess * self.sim_freq
        ).clamp(0.0, self._bead_force_max)

        force = torch.zeros((self.num_envs, 3), device=self.device)
        force[:, 2] = torch.where(engaged, resist, torch.zeros_like(resist))
        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            force_buf[self.cap._body_data_index, :3] = force
            self.scene.px.gpu_apply_rigid_dynamic_force()
        else:
            self.cap._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))

    def _bead_engagement(self) -> tuple[torch.Tensor, torch.Tensor]:
        throat_jar = self._cap_throat_in_jar_frame()
        depth = self._neck_top_local_z - throat_jar[:, 2]
        lateral = torch.linalg.norm(throat_jar[:, :2], dim=-1)
        return depth, lateral

    def _cap_throat_in_jar_frame(self) -> torch.Tensor:
        throat_world = self.cap.pose * self._cap_throat_offset
        return (self.jar.pose.inv() * throat_world).p

    def _reset_press_trackers(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_seat_sustain_counter"):
            self._seat_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._bead_cleared = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._bead_force_scale = torch.ones(self.num_envs, device=self.device)
            self._bead_prev_depth = torch.zeros(self.num_envs, device=self.device)
            self._bead_prev_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._seat_sustain_counter[env_idx] = 0
        self._bead_cleared[env_idx] = False
        self._bead_prev_depth[env_idx] = -1.0
        self._bead_prev_engaged[env_idx] = False
        self._bead_force_scale[env_idx] = self._sample_bead_interference(b)

    def _sample_bead_interference(self, b: int) -> torch.Tensor:
        if not self.randomization.bead_interference:
            return torch.ones(b, device=self.device)
        lo, hi = self._bead_interference_range
        scale = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(scale, device=self.device)

    def evaluate(self):
        depth, lateral = self._bead_engagement()
        seated = depth >= self._seat_success_depth
        centered = lateral < self._cap_lateral_threshold

        cap_axis = quaternion_to_matrix(self.cap.pose.q)[:, :, 2]
        jar_axis = quaternion_to_matrix(self.jar.pose.q)[:, :, 2]
        coaxial = (cap_axis * jar_axis).sum(dim=-1) > self._cap_axis_cos_threshold
        jar_upright = jar_axis[:, 2] > self._jar_upright_cos

        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        away = torch.linalg.norm(tcp_pos - self.cap.pose.p, dim=-1) > self._released_distance_threshold
        fingers_open = self.gripper_aperture() > self._released_aperture
        released = away & fingers_open

        rel_speed = torch.linalg.norm(self.cap.linear_velocity - self.jar.linear_velocity, dim=-1)
        comoving = rel_speed < self._comoving_rel_speed_threshold

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        jar_on_belt = rider_on_belt(
            pose=self.jar.pose,
            linear_velocity=self.jar.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._jar_body_radius,
            y_tol=self._conveyor_half_length + self._jar_body_radius,
            rest_height=self._jar_origin_above_deck,
            tilt_cos=self._jar_upright_cos,
        )

        pressed = seated & centered & coaxial & released & comoving & jar_upright & jar_on_belt

        self._seat_sustain_counter = self._advance_sustain("seat_sustain", self._seat_sustain_counter, pressed)
        success = self._seat_sustain_counter >= self._seat_sustain_steps

        return {
            "success": success,
            "seated": seated,
            "centered": centered,
            "coaxial": coaxial,
            "released": released,
            "comoving": comoving,
            "jar_upright": jar_upright,
            "jar_on_belt": jar_on_belt,
            "seat_depth": depth,
            "cap_lateral_offset": lateral,
            "press_sustain": self._seat_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        _depth, lateral = self._bead_engagement()

        align_reward = 1 - torch.tanh(20.0 * lateral)

        on_neck = (lateral < self._bead_engage_radius).float()
        progress = (info["seat_depth"] / self._seat_success_depth).clamp(0.0, 1.0) * on_neck

        withdraw = (
            torch.linalg.norm(tcp_pos - self.cap.pose.p, dim=-1).clamp(max=self._released_distance_threshold)
            / self._released_distance_threshold
        )
        return align_reward + 2.0 * progress + 2.0 * (info["seated"] & info["centered"]).float() * withdraw

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
