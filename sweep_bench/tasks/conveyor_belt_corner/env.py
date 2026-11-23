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

from sweep_bench.assets.conveyor import build_conveyor, step_belts
from sweep_bench.assets.procedural import build_box
from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_belt_corner.config import RandomizationConfig
from sweep_bench.tasks.conveyor_belt_corner.scripted import ScriptedBeltCornerMixin
from sweep_bench.tasks.conveyor_belt_corner.variants import BELT_CORNER_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.conveyor import ConveyorBelt

_DISTRACTOR_COLOR = (0.45, 0.50, 0.42, 1.0)


@bench_task("ConveyorBeltCorner-v0")
@register_env("ConveyorBeltCorner-v0", max_episode_steps=300)
class ConveyorBeltCornerEnv(ScriptedBeltCornerMixin, ConveyorWorkspaceEnv):
    """Transfer a part from the main belt onto a BRANCH belt at 90 degrees."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(BELT_CORNER_OBJECT_IDS)

    _belt_b_dx: float = -0.53
    _belt_b_dy_mag: float = 0.34
    _belt_b_half_length: float = 0.40
    _belt_b_yaw: float = math.pi / 2

    _place_x_default: float = -0.24

    _part_patch_radius: float = 0.02

    _spawn_margin: float = 0.04

    _distractor_pitch: float = 0.30
    _distractor_half: float = 0.02
    _distractor_patch_radius: float = 0.02

    _min_part_half_z: float = 0.015
    _max_part_half_fp: float = 0.035

    _deck_rest_tol: float = 0.010
    _belt_b_band_half: float = 0.045
    _tip_max_deg: float = 15.0
    _comove_tol: float = 0.06
    _lateral_speed_tol: float = 0.05
    _part_spin_tol: float = 0.20
    _part_rise_tol: float = 0.05
    _release_finger_margin: float = 0.01
    _released_distance: float = 0.06
    _tcp_clear_above_deck: float = 0.12
    _belt_b_run_x: float = 0.35
    _corner_sustain_steps: int = 4

    _gap_rest_tol: float = 0.01
    _rider_z_tol: float = 0.006
    _rider_x_tol: float = 0.05
    _rider_tilt_deg: float = 10.0

    default_prompt: str = "move the part from the main conveyor onto the branch conveyor"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        shortest = min(spec_by_id[o].bbox[2] for o in self.objects)
        widest = max(max(spec_by_id[o].bbox[0], spec_by_id[o].bbox[1]) for o in self.objects)
        if shortest < self._min_part_half_z:
            raise ValueError(
                f"part half-height {shortest} m is below {self._min_part_half_z} m: the deposit would drive "
                f"the fingertips into the branch deck before the release fires"
            )
        if widest > self._max_part_half_fp:
            raise ValueError(
                f"part footprint half-extent {widest} m exceeds {self._max_part_half_fp} m: it could satisfy "
                f"the branch band's centre test with its footprint off the deck"
            )

    _human_render_eye: tuple[float, float, float] = (0.90, 0.10, 1.35)
    _human_render_target: tuple[float, float, float] = (-0.20, -0.12, 0.28)

    @property
    def _default_human_render_camera_configs(self):
        mirror = -self._branch_side
        ex, ey, ez = self._human_render_eye
        tx, ty, tz = self._human_render_target
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=(ex, mirror * ey, ez),
            target=(tx, mirror * ty, tz),
        )

    @property
    def _branch_side(self) -> float:
        return getattr(self, "_branch_side_sign", -1.0)

    @property
    def _belt_b_dy(self) -> float:
        return self._branch_side * self._belt_b_dy_mag

    @property
    def _branch_center(self) -> tuple[float, float]:
        ox, oy, _oz = self.workspace_offset
        return ox + self._belt_b_dx, oy + self._belt_b_dy

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        device = self.device

        side = -1.0
        if self.randomization.branch_side:
            side = float(self._episode_rng.choice([-1.0, 1.0]))
        self._branch_side_sign = side

        bx, by = ox + self._belt_b_dx, oy + self._belt_b_dy
        self.belt_b = build_conveyor(
            self.scene,
            workspace_offset=(bx, by, oz),
            deck_height=self._conveyor_deck_height,
            belt_half_width=self._conveyor_half_width,
            half_length=self._belt_b_half_length,
            static_friction=self.belt_static_friction,
            kinetic_friction=self.belt_kinetic_friction,
            name="conveyor_b",
            yaw=self._belt_b_yaw,
            texture_slot=1,
        )

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        part_objs: list[Actor] = []
        patch_radii: list[float] = []
        for i, name in enumerate(self._part_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, oz + self._conveyor_deck_height + 0.2 + spec.bbox[2]])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            part_objs.append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._part_patch_radius)
        self.part = Actor.merge(part_objs, name="transfer_part")
        self.add_to_state_dict_registry(self.part)

        patch = torch.as_tensor(patch_radii, dtype=torch.float32, device=device)
        self.belt.register_rider(self.part, patch_radius=patch)
        self.belt_b.register_rider(self.part, patch_radius=patch)

        specs = [spec_by_id[name] for name in self._part_names]

        def _tensor(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=device)

        self._part_half_z = _tensor([s.bbox[2] for s in specs])
        self._part_half_fp = _tensor([max(s.bbox[0], s.bbox[1]) for s in specs])
        self._part_grasp_width = _tensor(
            [s.grasp_width if s.grasp_width is not None else 2.0 * s.bbox[0] for s in specs]
        )

        self.distractor: Actor | None = None
        if self.randomization.distractor:
            half = self._distractor_half
            self.distractor = build_box(
                self.scene,
                half_x=half,
                half_y=half,
                half_z=half,
                color=_DISTRACTOR_COLOR,
                initial_pose=sapien.Pose(p=[ox, oy, oz + self._conveyor_deck_height + half]),
            ).build(name="belt_a_distractor")
            self.belt.register_rider(self.distractor, patch_radius=self._distractor_patch_radius)

        self._release_xy = torch.zeros((self.num_envs, 2), device=device)

    @property
    def _velocity_belts(self) -> list[ConveyorBelt]:
        return [self.belt, self.belt_b]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        self.belt_b.reset(env_idx, self._setup_branch_motion(env_idx, b))

        sign = self.belt.direction_sign[env_idx]

        pos = self._sample_part_position(env_idx, b)
        self.part.set_pose(Pose.create_from_pq(pos, self._sample_part_yaw(b)))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.part.set_linear_velocity(vel)
        self.part.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.distractor is not None:
            d_pos = torch.zeros((b, 3), device=device)
            d_pos[:, 0] = pos[:, 0]
            d_pos[:, 1] = pos[:, 1] + self._distractor_pitch * sign
            d_pos[:, 2] = deck_z + self._distractor_half + 0.001
            d_q = torch.zeros((b, 4), device=device)
            d_q[:, 0] = 1.0
            self.distractor.set_pose(Pose.create_from_pq(d_pos, d_q))
            self.distractor.set_linear_velocity(vel)
            self.distractor.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._release_xy[env_idx] = self._sample_release_xy(b)

        self._reset_corner_trackers(env_idx)
        if self.distractor is not None:
            self._distractor_spawn_xy[env_idx] = d_pos[:, :2]
            self._distractor_spawn_q[env_idx] = d_q

    def _setup_branch_motion(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        rcfg = self.randomization
        if rcfg.belt_b_speed:
            lo, hi = rcfg.belt_b_speed_range
            speeds = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            return torch.as_tensor(speeds, device=self.device)
        return torch.abs(self.belt.velocity[env_idx])

    def _sample_part_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        start_y = self._conveyor_half_length - self._part_half_fp[env_idx] - self._spawn_margin
        pos = torch.zeros((b, 3), device=self.device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = oz + self._conveyor_deck_height + self._part_half_z[env_idx] + 0.001
        return pos

    def _sample_part_yaw(self, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        if self.randomization.rider_yaw:
            lo, hi = self.randomization.rider_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_release_xy(self, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        xy = torch.zeros((b, 2), device=self.device)
        if self.randomization.place_x:
            lo, hi = self.randomization.place_x_range
            xs = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            xy[:, 0] = ox + torch.as_tensor(xs, device=self.device)
        else:
            xy[:, 0] = ox + self._place_x_default
        xy[:, 1] = oy + self._belt_b_dy
        return xy

    def _before_control_step(self) -> None:
        super()._before_control_step()
        self.belt_b.on_control_step()

    def _before_simulation_step(self) -> None:
        step_belts(self.belt, self.belt_b)

    def _reset_corner_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_corner_sustain"):
            self._corner_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._delivered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._distractor_disturbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._distractor_spawn_xy = torch.zeros((self.num_envs, 2), device=self.device)
            self._distractor_spawn_q = torch.zeros((self.num_envs, 4), device=self.device)
            self._distractor_spawn_q[:, 0] = 1.0
            self._gap_rested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._part_spawn_up = torch.zeros((self.num_envs, 3), device=self.device)
            self._part_spawn_up[:, 2] = 1.0
        self._corner_sustain[env_idx] = 0
        self._delivered[env_idx] = False
        self._distractor_disturbed[env_idx] = False
        self._gap_rested[env_idx] = False
        self._part_spawn_up[env_idx] = quaternion_to_matrix(self.part.pose.q)[env_idx, :, 2]

    def _decoy_disturbed_now(self, decoy_pose: Pose, deck_z: float) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        d_pos = decoy_pose.p
        d_lifted = torch.abs(d_pos[:, 2] - (deck_z + self._distractor_half)) > self._rider_z_tol
        d_shoved = torch.abs(d_pos[:, 0] - ox) > self._rider_x_tol
        d_on_belt = torch.abs(d_pos[:, 1] - oy) <= self._conveyor_half_length - self._distractor_half
        return (d_lifted | d_shoved | self._decoy_orientation_disturbed(decoy_pose.q)) & d_on_belt

    def _part_settled(self) -> torch.Tensor:
        return (torch.linalg.norm(self.part.angular_velocity, dim=-1) <= self._part_spin_tol) & (
            torch.abs(self.part.linear_velocity[:, 2]) <= self._part_rise_tol
        )

    def _decoy_orientation_disturbed(self, decoy_q: torch.Tensor) -> torch.Tensor:
        return quaternion_to_matrix(decoy_q)[:, 2, 2] < math.cos(math.radians(self._rider_tilt_deg))

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        bx, by = self._branch_center
        part_pos = self.part.pose.p
        part_vel = self.part.linear_velocity
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        v_b = self.belt_b.velocity

        seated = torch.abs(part_pos[:, 2] - (deck_z + self._part_half_z)) <= self._deck_rest_tol
        in_band = torch.abs(part_pos[:, 1] - by) <= self._belt_b_band_half

        up_now = quaternion_to_matrix(self.part.pose.q)[:, :, 2]
        tip_cos = (up_now * self._part_spawn_up).sum(dim=-1)
        not_tipped = tip_cos >= math.cos(math.radians(self._tip_max_deg))

        along = torch.abs(part_vel[:, 0] - (-v_b)) <= self._comove_tol
        cross_still = torch.abs(part_vel[:, 1]) <= self._lateral_speed_tol
        comoving = along & cross_still & self._part_settled()

        finger_width = self.gripper_aperture()
        fingers_open = finger_width >= self._part_grasp_width + self._release_finger_margin
        hand_clear = torch.linalg.norm(tcp_pos - part_pos, dim=-1) > self._released_distance
        released = fingers_open & hand_clear

        tcp_clear = (tcp_pos[:, 2] - deck_z) >= self._tcp_clear_above_deck

        on_branch_run = torch.abs(part_pos[:, 0] - bx) <= self._belt_b_run_x
        branch_travelling = (-part_vel[:, 0]) * v_b >= 0.5 * v_b * v_b

        riding = seated & in_band & not_tipped & comoving & released & tcp_clear & on_branch_run & branch_travelling
        self._corner_sustain = self._advance_sustain("belt_corner_riding", self._corner_sustain, riding)

        if self.distractor is not None:
            self._distractor_disturbed = self._distractor_disturbed | (
                self._decoy_disturbed_now(self.distractor.pose, deck_z) & ~self._delivered
            )
            distractor_slip = torch.abs(self.distractor.linear_velocity[:, 1] - self.belt.velocity)
        else:
            distractor_slip = torch.zeros(self.num_envs, device=self.device)
        self._delivered = self._delivered | (self._corner_sustain >= self._corner_sustain_steps)

        underside = part_pos[:, 2] - self._part_half_z
        on_surface = torch.abs(underside - oz) <= self._gap_rest_tol
        off_belt_a = (torch.abs(part_pos[:, 0] - ox) > self._conveyor_half_width) | (
            torch.abs(part_pos[:, 1] - oy) > self._conveyor_half_length
        )
        off_belt_b = (torch.abs(part_pos[:, 1] - by) > self._conveyor_half_width) | (
            torch.abs(part_pos[:, 0] - bx) > self._belt_b_half_length
        )
        self._gap_rested = self._gap_rested | (on_surface & off_belt_a & off_belt_b)

        success = (self._corner_sustain >= self._corner_sustain_steps) & ~self._distractor_disturbed & ~self._gap_rested

        return {
            "success": success,
            "seated": seated,
            "in_band": in_band,
            "not_tipped": not_tipped,
            "comoving": comoving,
            "along_branch": along,
            "cross_still": cross_still,
            "released": released,
            "fingers_open": fingers_open,
            "hand_clear": hand_clear,
            "tcp_clear": tcp_clear,
            "on_branch_run": on_branch_run,
            "branch_travelling": branch_travelling,
            "riding": riding,
            "corner_sustain": self._corner_sustain,
            "delivered": self._delivered,
            "distractor_disturbed": self._distractor_disturbed,
            "gap_rested": self._gap_rested,
            "distractor_along_slip": distractor_slip,
            "part_to_release_xy": torch.linalg.norm(part_pos[:, :2] - self._release_xy, dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        _ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        _bx, by = self._branch_center
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - part_pos, dim=-1))

        to_branch = torch.stack(
            [part_pos[:, 1] - by, part_pos[:, 2] - (deck_z + self._part_half_z)],
            dim=-1,
        )
        transfer_reward = 1 - torch.tanh(3 * torch.linalg.norm(to_branch, dim=-1))

        clean = (~info["distractor_disturbed"] & ~info["gap_rested"]).float()
        return reach_reward + transfer_reward + 2.0 * info["riding"].float() * clean

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"move the {name} from the main conveyor onto the branch conveyor" for name in self._part_display]
