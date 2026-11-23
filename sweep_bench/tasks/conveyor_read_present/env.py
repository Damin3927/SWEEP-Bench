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

from sweep_bench.assets.vetted import resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_read_present.assets import build_reader_window
from sweep_bench.tasks.conveyor_read_present.config import RandomizationConfig
from sweep_bench.tasks.conveyor_read_present.scripted import ScriptedReadPresentMixin
from sweep_bench.tasks.conveyor_read_present.variants import (
    MARKED_PART_SPECS,
    READ_PRESENT_OBJECT_IDS,
    mark_color_name,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorReadPresent-v0")
@register_env("ConveyorReadPresent-v0", max_episode_steps=300)
class ConveyorReadPresentEnv(ScriptedReadPresentMixin, ConveyorWorkspaceEnv):
    """Present a part's marked face to a fixed reader, then put it back on the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(READ_PRESENT_OBJECT_IDS, extra=MARKED_PART_SPECS)

    _read_dx: float = -0.24
    _read_dy: float = -0.30
    _read_dz: float = 0.16
    _read_box_half: float = 0.025
    _reader_yaw_deg: float = 35.0
    _read_box_dist: float = 0.13
    _window_half: float = 0.045
    _frame_bar_half: float = 0.0065
    _frame_half_thickness: float = 0.008
    _post_half_width: float = 0.022
    _reader_post_slack: float = 0.03

    _part_patch_radius: float = 0.020
    _place_dy: float = -0.35
    _present_tcp_dz: float = 0.0

    _read_cone_deg: float = 35.0
    _read_dwell_steps: int = 12
    _deck_rest_tol: float = 0.006
    _band_half: float = 0.045
    _tip_max_deg: float = 15.0
    _released_finger_width: float = 0.072
    _released_distance: float = 0.06
    _comove_tol: float = 0.06
    _lateral_speed_tol: float = 0.05
    _runway_min: float = 0.30
    _drop_below: float = 0.02
    _present_sustain_steps: int = 12

    default_prompt: str = "hold the marked face of the part up to the reader window, then set it back on the belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        jitter = max(abs(v) for v in self.randomization.mark_face_range)
        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        gripped = 2.0 * max(
            spec_by_id[o].bbox[1] * math.cos(jitter) + spec_by_id[o].bbox[0] * math.sin(jitter) for o in self.objects
        )
        if gripped >= self._released_finger_width:
            raise ValueError(
                f"a gripped part spans {gripped:.4f} m, which is not below the released finger opening "
                f"{self._released_finger_width} m: a gripping hand would read as released"
            )

    _human_render_eye: tuple[float, float, float] = (0.90, 0.00, 1.05)
    _human_render_target: tuple[float, float, float] = (-0.10, 0.00, 0.36)

    @property
    def _default_human_render_camera_configs(self):
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=self._human_render_eye,
            target=self._human_render_target,
        )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        device = self.device
        deck_z = oz + self._conveyor_deck_height

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_marks: list[str] = [mark_color_name(spec_by_id[name]) for name in self._part_names]
        part_objs: list[Actor] = []
        patch_radii: list[float] = []
        halves: list[tuple[float, float, float]] = []
        for i, name in enumerate(self._part_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            obj.angular_damping = 5.0
            part_objs.append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._part_patch_radius)
            halves.append(spec.bbox)
        self.part = Actor.merge(part_objs, name="marked_part")
        self.add_to_state_dict_registry(self.part)
        self.belt.register_rider(
            self.part,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=device),
        )
        self._part_half = torch.as_tensor(halves, dtype=torch.float32, device=device)

        reader_builder = build_reader_window(
            self.scene,
            post_height=self._conveyor_deck_height + self._read_dz + self._reader_post_slack,
            window_half=self._window_half,
            frame_bar_half=self._frame_bar_half,
            frame_half_thickness=self._frame_half_thickness,
            post_half_width=self._post_half_width,
        )
        reader_builder.initial_pose = sapien.Pose(p=[ox + self._read_dx, oy, deck_z + self._read_dz])
        self.reader = reader_builder.build_kinematic(name="reader_window")

        self._read_centre = torch.zeros((self.num_envs, 3), device=device)
        self._reader_look = torch.zeros((self.num_envs, 3), device=device)
        self._reader_look[:, 0] = 1.0
        self._target_heading = torch.zeros((self.num_envs,), device=device)
        self._present_xyz = torch.zeros((self.num_envs, 3), device=device)
        self._place_xy = torch.zeros((self.num_envs, 2), device=device)
        self._part_spawn_up = torch.zeros((self.num_envs, 3), device=device)

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"hold the {mark}-marked face of the part up to the reader window, then set it back on the belt"
            for mark in self._part_marks
        ]

    def _reader_frame(self, env_idx: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        ox, oy, oz = self.workspace_offset
        device = self.device
        sign = self.belt.direction_sign[env_idx]
        deck_z = oz + self._conveyor_deck_height

        dx = torch.full((b,), self._read_dx, device=device)
        dy = torch.full((b,), self._read_dy, device=device)
        dz = torch.full((b,), self._read_dz, device=device)
        if self.randomization.reader_pose:
            cfg = self.randomization
            dx = dx + torch.as_tensor(
                self._episode_rng.uniform(*cfg.reader_dx_range, size=b).astype(np.float32), device=device
            )
            dy = dy + torch.as_tensor(
                self._episode_rng.uniform(*cfg.reader_dy_range, size=b).astype(np.float32), device=device
            )
            dz = dz + torch.as_tensor(
                self._episode_rng.uniform(*cfg.reader_dz_range, size=b).astype(np.float32), device=device
            )

        centre = torch.stack([ox + dx, oy + sign * dy, deck_z + dz], dim=-1)
        phi = math.radians(self._reader_yaw_deg)
        look = torch.zeros((b, 3), device=device)
        look[:, 0] = math.cos(phi)
        look[:, 1] = sign * math.sin(phi)
        return centre, look

    def _sample_part_yaw(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        sign = self.belt.direction_sign[env_idx]
        yaw = -sign * (math.pi / 2)
        if self.randomization.mark_face:
            lo, hi = self.randomization.mark_face_range
            jitter = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            yaw = yaw + torch.as_tensor(jitter, device=device)
        return yaw

    def _sample_part_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        half = self._part_half[env_idx]
        half_fp = torch.linalg.norm(half[:, :2], dim=-1)
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - (self._conveyor_half_length - half_fp - 0.04) * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + half[:, 2] + 0.001
        return pos

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.reader]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        oy = self.workspace_offset[1]
        sign = self.belt.direction_sign[env_idx]

        centre, look = self._reader_frame(env_idx, b)
        self._read_centre[env_idx] = centre
        self._reader_look[env_idx] = look
        aperture_p = centre - self._read_box_dist * look
        yaw = torch.atan2(look[:, 1], look[:, 0])
        aperture_q = torch.zeros((b, 4), device=device)
        aperture_q[:, 0] = torch.cos(0.5 * yaw)
        aperture_q[:, 3] = torch.sin(0.5 * yaw)
        self.reader.set_pose(Pose.create_from_pq(aperture_p, aperture_q))

        self._target_heading[env_idx] = torch.atan2(-look[:, 1], -look[:, 0])
        self._present_xyz[env_idx] = centre + torch.tensor([0.0, 0.0, self._present_tcp_dz], device=device)
        self._place_xy[env_idx, 0] = self.workspace_offset[0]
        self._place_xy[env_idx, 1] = oy + sign * self._place_dy

        part_pos = self._sample_part_position(env_idx, b)
        part_yaw = self._sample_part_yaw(env_idx, b)
        part_q = torch.zeros((b, 4), device=device)
        part_q[:, 0] = torch.cos(0.5 * part_yaw)
        part_q[:, 3] = torch.sin(0.5 * part_yaw)
        self.part.set_pose(Pose.create_from_pq(part_pos, part_q))
        part_vel = torch.zeros((b, 3), device=device)
        part_vel[:, 1] = self.belt.velocity[env_idx]
        self.part.set_linear_velocity(part_vel)
        self.part.set_angular_velocity(torch.zeros((b, 3), device=device))
        self._part_spawn_up[env_idx] = quaternion_to_matrix(part_q)[:, :, 2]

        self._reset_read_present_trackers(env_idx)

    def _reset_read_present_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_read_dwell"):
            self._read_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._read_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._present_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._read_dwell[env_idx] = 0
        self._read_latched[env_idx] = False
        self._dropped[env_idx] = False
        self._present_sustain[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        part_pose = self.part.pose
        part_pos = part_pose.p
        part_vel = self.part.linear_velocity
        part_rot = quaternion_to_matrix(part_pose.q)
        tcp_pos = agent.tcp.pose.p
        look = self._reader_look

        across = torch.stack([-look[:, 1], look[:, 0], torch.zeros_like(look[:, 0])], dim=-1)
        rel = part_pos - self._read_centre
        in_read_box = (
            (torch.abs((rel * look).sum(dim=-1)) <= self._read_box_half)
            & (torch.abs((rel * across).sum(dim=-1)) <= self._read_box_half)
            & (torch.abs(rel[:, 2]) <= self._read_box_half)
        )
        mark_cos = -(part_rot[:, :, 0] * look).sum(dim=-1)
        facing_reader = mark_cos >= math.cos(math.radians(self._read_cone_deg))
        reading = in_read_box & facing_reader
        self._read_dwell = self._advance_sustain("read_present_dwell", self._read_dwell, reading)
        self._read_latched = self._read_latched | (self._read_dwell >= self._read_dwell_steps)

        on_deck = torch.abs(part_pos[:, 2] - (deck_z + self._part_half[:, 2])) <= self._deck_rest_tol

        in_band = torch.abs(part_pos[:, 0] - ox) <= self._band_half

        untipped = (part_rot[:, :, 2] * self._part_spawn_up).sum(dim=-1) >= math.cos(math.radians(self._tip_max_deg))

        finger_width = self.gripper_aperture()
        fingers_open = finger_width >= self._released_finger_width
        hand_away = torch.linalg.norm(tcp_pos - part_pos, dim=-1) > self._released_distance
        released = fingers_open & hand_away

        comoving = (torch.abs(part_vel[:, 1] - self.belt.velocity) <= self._comove_tol) & (
            torch.abs(part_vel[:, 0]) <= self._lateral_speed_tol
        )

        runway = self._conveyor_half_length - (part_pos[:, 1] - oy) * self.belt.direction_sign
        has_runway = runway >= self._runway_min

        self._dropped = self._dropped | (part_pos[:, 2] < deck_z - self._drop_below)

        returned = self._read_latched & on_deck & in_band & untipped & released & comoving & has_runway
        self._present_sustain = self._advance_sustain("read_present_hold", self._present_sustain, returned)
        success = (self._present_sustain >= self._present_sustain_steps) & ~self._dropped

        return {
            "success": success,
            "in_read_box": in_read_box,
            "facing_reader": facing_reader,
            "reading": reading,
            "read_dwell": self._read_dwell,
            "read_latched": self._read_latched,
            "on_deck": on_deck,
            "in_band": in_band,
            "untipped": untipped,
            "fingers_open": fingers_open,
            "hand_away": hand_away,
            "released": released,
            "comoving": comoving,
            "has_runway": has_runway,
            "dropped": self._dropped,
            "returned": returned,
            "present_sustain": self._present_sustain,
            "mark_error_deg": torch.rad2deg(torch.arccos(mark_cos.clamp(-1.0, 1.0))),
            "part_to_read_centre": torch.linalg.norm(rel, dim=-1),
            "runway": runway,
        }

    def compute_dense_reward(self, obs, action, info):
        ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - part_pos, dim=-1))
        present = 1 - torch.tanh(3 * torch.linalg.norm(part_pos - self._read_centre, dim=-1))
        aim = 1 - torch.clamp(info["mark_error_deg"] / 180.0, 0, 1)
        read_reward = torch.where(info["read_latched"], torch.ones_like(present), 0.5 * (present + aim))
        to_belt = torch.stack([part_pos[:, 0] - ox, part_pos[:, 2] - (deck_z + self._part_half[:, 2])], dim=-1)
        place_reward = 1 - torch.tanh(3 * torch.linalg.norm(to_belt, dim=-1))
        riding = info["returned"].float()
        clean = (~info["dropped"]).float()

        return reach_reward + 2.0 * read_reward + place_reward + 2.0 * riding * clean

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0
