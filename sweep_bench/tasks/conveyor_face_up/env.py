from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_face_up.assets import build_marked_die, build_place_pad, face_normal_world
from sweep_bench.tasks.conveyor_face_up.config import RandomizationConfig
from sweep_bench.tasks.conveyor_face_up.scripted import ScriptedFaceUpMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_DIE_BODY_COLOR = (0.92, 0.90, 0.86, 1.0)
_FACE_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.80, 0.16, 0.16, 1.0),
    (0.16, 0.62, 0.24, 1.0),
    (0.18, 0.36, 0.82, 1.0),
    (0.92, 0.74, 0.10, 1.0),
    (0.92, 0.45, 0.10, 1.0),
    (0.52, 0.24, 0.72, 1.0),
)
_FACE_NAMES: tuple[str, ...] = ("red", "green", "blue", "yellow", "orange", "purple")

_PAD_COLOR = (0.34, 0.37, 0.42, 1.0)
_PAD_PAINT_COLOR = (0.85, 0.85, 0.88, 1.0)


def _cube_rotation_quats() -> np.ndarray:
    mats: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3), dtype=np.float64)
            for col, row in enumerate(perm):
                m[row, col] = signs[col]
            if np.linalg.det(m) > 0.5:
                mats.append(m)
    quats = matrix_to_quaternion(torch.as_tensor(np.stack(mats), dtype=torch.float32))
    return quats.numpy().astype(np.float32)


_CUBE_ROTATIONS = _cube_rotation_quats()


@bench_task("ConveyorFaceUp-v0")
@register_env("ConveyorFaceUp-v0", max_episode_steps=300)
class ConveyorFaceUpEnv(ScriptedFaceUpMixin, ConveyorWorkspaceEnv):
    """Pick a die off the belt and SET IT DOWN showing the commanded face up."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _die_half: float = 0.0225
    _die_patch_radius: float = 0.0225

    _pad_half_x: float = 0.07
    _pad_half_y: float = 0.07
    _pad_paint_half_x: float = 0.04
    _pad_paint_half_y: float = 0.04
    _pad_offset_x: float = -0.28
    _pad_offset_y_mag: float = 0.30
    _pad_fast_band_speed: float = 0.25

    _face_up_tol_deg: float = 15.0
    _rest_height_tol: float = 0.004
    _settled_speed: float = 0.02
    _settled_ang_speed: float = 0.30
    _pad_footprint_inset: float = 0.010
    _jaws_open_gap: float = 0.060
    _hand_clear_dist: float = 0.10
    _face_up_sustain_steps: int = 10

    _human_render_eye: tuple[float, float, float] = (0.75, 0.0, 1.25)
    _human_render_target: tuple[float, float, float] = (-0.1, 0.0, 0.25)

    default_prompt: str = "set the die down on the pad with the commanded face up"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(self, *args, **kwargs) -> None:
        near_pad_x = abs(self._pad_offset_x) - self._pad_half_x
        if near_pad_x <= self._conveyor_half_width + self._die_half:
            raise ValueError(
                f"pad footprint reaches |x - ox| = {near_pad_x:.3f} m, inside the belt band + die half "
                f"({self._conveyor_half_width + self._die_half:.3f} m): the painted-footprint success term "
                f"would no longer imply the off-belt guard"
            )
        super().__init__(*args, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=self._human_render_eye,
            target=self._human_render_target,
        )

    @property
    def _pad_half_z(self) -> float:
        return 0.5 * self._conveyor_deck_height

    @property
    def episode_prompts(self) -> list[str]:
        return [f"set the die down on the pad with its {_FACE_NAMES[int(f)]} face up" for f in self._commanded_face]

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        die_builder = build_marked_die(
            self.scene,
            half=self._die_half,
            body_color=_DIE_BODY_COLOR,
            face_colors=_FACE_COLORS,
        )
        die_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._die_half + 0.001])
        self.die = die_builder.build(name="marked_die")
        self.belt.register_rider(self.die, patch_radius=self._die_patch_radius)

        pad_builder = build_place_pad(
            self.scene,
            half_x=self._pad_half_x,
            half_y=self._pad_half_y,
            half_z=self._pad_half_z,
            paint_half_x=self._pad_paint_half_x,
            paint_half_y=self._pad_paint_half_y,
            color=_PAD_COLOR,
            paint_color=_PAD_PAINT_COLOR,
        )
        pad_builder.initial_pose = sapien.Pose(
            p=[ox + self._pad_offset_x, oy + self._pad_offset_y_mag, oz + self._pad_half_z]
        )
        self.pad = pad_builder.build_kinematic(name="place_pad")

        self._commanded_face = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command_sign = torch.ones(self.num_envs, device=self.device)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.pad]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        pad_pos = self._sample_pad_position(env_idx, b)
        pad_q = torch.zeros((b, 4), device=device)
        pad_q[:, 0] = 1.0
        self.pad.set_pose(Pose.create_from_pq(pad_pos, pad_q))

        if self.randomization.commanded_face:
            drawn = self._episode_rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=b)
            command_sign = torch.as_tensor(np.asarray(drawn, dtype=np.float32), device=device)
        else:
            command_sign = torch.ones(b, device=device)
        self._command_sign[env_idx] = command_sign

        die_pos = self._sample_die_position(env_idx, b)
        die_q = self._sample_die_orientation(b)
        self.die.set_pose(Pose.create_from_pq(die_pos, die_q))
        die_vel = torch.zeros((b, 3), device=device)
        die_vel[:, 1] = self.belt.velocity[env_idx]
        self.die.set_linear_velocity(die_vel)
        self.die.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._commanded_face[env_idx] = self._sample_commanded_face(die_q, command_sign)
        self._reset_face_up_trackers(env_idx)

    def _sample_die_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._die_half - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._die_half + 0.001
        return pos

    def _sample_die_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if not self.randomization.spawn_roll:
            return q
        idx = self._episode_rng.choice(np.arange(len(_CUBE_ROTATIONS)), size=b)
        return torch.as_tensor(_CUBE_ROTATIONS[np.asarray(idx)], dtype=torch.float32, device=device)

    def _sample_commanded_face(self, die_q: torch.Tensor, command_sign: torch.Tensor) -> torch.Tensor:
        rot = quaternion_to_matrix(die_q)
        in_body = rot[:, 1, :] * command_sign.unsqueeze(-1)
        axis = in_body.abs().argmax(dim=-1)
        picked = torch.gather(in_body, 1, axis.unsqueeze(-1)).squeeze(-1)
        return 2 * axis + (picked < 0).long()

    def _sample_pad_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox + self._pad_offset_x
        if self.randomization.pad_side:
            side_np = self._episode_rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=b)
            side = torch.as_tensor(np.asarray(side_np, dtype=np.float32), device=device)
        else:
            speed = self.belt.velocity[env_idx].abs()
            fast_band = (speed >= self._pad_fast_band_speed) | (speed == 0.0)
            grasp_rel = torch.where(fast_band, 1.0, -1.0)
            side = grasp_rel * self.belt.direction_sign[env_idx]
        pos[:, 1] = oy + self._pad_offset_y_mag * side
        pos[:, 2] = oz + self._pad_half_z
        return pos

    def _reset_face_up_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_face_up_sustain"):
            self._face_up_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._prev_die_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_die_q = torch.zeros((self.num_envs, 4), device=self.device)
        self._face_up_sustain[env_idx] = 0
        self._prev_die_pos[env_idx] = self.die.pose.p[env_idx]
        self._prev_die_q[env_idx] = self.die.pose.q[env_idx]

    def evaluate(self):
        ox = self.workspace_offset[0]
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        die_pos = self.die.pose.p
        die_q = self.die.pose.q
        pad_pos = self.pad.pose.p

        advanced = self._eval_advanced("face_up_pose").unsqueeze(-1)
        lin_speed = torch.linalg.norm((die_pos - self._prev_die_pos) * self.control_freq, dim=-1)
        q_dot = (die_q * self._prev_die_q).sum(dim=-1).abs().clamp(max=1.0)
        ang_speed = 2.0 * torch.arccos(q_dot) * self.control_freq
        self._prev_die_pos = torch.where(advanced, die_pos, self._prev_die_pos)
        self._prev_die_q = torch.where(advanced, die_q, self._prev_die_q)

        face_cos = face_normal_world(die_q, self._commanded_face)[:, 2]
        faces_up = face_cos >= math.cos(math.radians(self._face_up_tol_deg))

        pad_top = pad_pos[:, 2] + self._pad_half_z
        die_bottom = die_pos[:, 2] - self._die_half
        on_pad = torch.abs(die_bottom - pad_top) < self._rest_height_tol

        not_sliding = lin_speed < self._settled_speed
        not_turning = ang_speed < self._settled_ang_speed

        rel = die_pos[:, :2] - pad_pos[:, :2]
        on_footprint = (torch.abs(rel[:, 0]) < self._pad_paint_half_x - self._pad_footprint_inset) & (
            torch.abs(rel[:, 1]) < self._pad_paint_half_y - self._pad_footprint_inset
        )

        finger_gap = self.gripper_aperture()
        jaws_open = finger_gap > self._jaws_open_gap
        hand_clear = torch.linalg.norm(tcp_pos - die_pos, dim=-1) > self._hand_clear_dist

        holding = faces_up & on_pad & not_sliding & not_turning & on_footprint & jaws_open & hand_clear
        self._face_up_sustain = self._advance_sustain("face_up_sustain", self._face_up_sustain, holding)

        return {
            "success": self._face_up_sustain >= self._face_up_sustain_steps,
            "faces_up": faces_up,
            "on_pad": on_pad,
            "not_sliding": not_sliding,
            "not_turning": not_turning,
            "on_footprint": on_footprint,
            "jaws_open": jaws_open,
            "hand_clear": hand_clear,
            "face_up_sustain": self._face_up_sustain,
            "face_error_deg": torch.rad2deg(torch.arccos(face_cos.clamp(-1.0, 1.0))),
            "die_speed": lin_speed,
            "die_ang_speed": ang_speed,
            "finger_gap": finger_gap,
            "off_belt": torch.abs(die_pos[:, 0] - ox) > self._conveyor_half_width + self._die_half,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        die_pos = self.die.pose.p
        pad_pos = self.pad.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - die_pos, dim=-1))
        carry_reward = 1 - torch.tanh(3 * torch.linalg.norm(die_pos[:, :2] - pad_pos[:, :2], dim=-1))
        face_reward = 1 - torch.clamp(info["face_error_deg"] / 180.0, 0, 1)
        placed = (info["on_pad"] & info["on_footprint"] & info["jaws_open"] & info["hand_clear"]).float()
        return reach_reward + carry_reward + 2.0 * face_reward + 2.0 * placed

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0
