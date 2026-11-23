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
from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_belt_swap.config import RandomizationConfig
from sweep_bench.tasks.conveyor_belt_swap.scripted import ScriptedBeltSwapMixin
from sweep_bench.tasks.conveyor_belt_swap.variants import BELT_SWAP_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.conveyor import ConveyorBelt


@bench_task("ConveyorBeltSwap-v0")
@register_env("ConveyorBeltSwap-v0", max_episode_steps=300)
class ConveyorBeltSwapEnv(ScriptedBeltSwapMixin, ConveyorWorkspaceEnv):
    """Transfer a part from one belt onto a parallel belt running the OTHER way."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(BELT_SWAP_OBJECT_IDS)

    _belt_b_dx: float = -0.26
    _belt_b_speed_ratio: float = 1.0

    _part_patch_radius: float = 0.02
    _distractor_pitch: float = 0.35

    _finger_below_tcp: float = 0.009
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
    _belt_b_run_y: float = 0.85
    _swap_sustain_steps: int = 4

    _rider_z_tol: float = 0.006
    _rider_x_tol: float = 0.05
    _rider_tilt_deg: float = 10.0
    _gap_rest_tol: float = 0.01

    default_prompt: str = "move the part from the far conveyor belt onto the near conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        shortest = min(spec_by_id[o].bbox[2] for o in self.objects)
        widest = max(max(spec_by_id[o].bbox[0], spec_by_id[o].bbox[1]) for o in self.objects)
        if shortest < self._finger_below_tcp:
            raise ValueError(
                f"part half-height {shortest} m is below the fingertip drop under the TCP "
                f"({self._finger_below_tcp} m): the grasp would drive the fingertips into the deck"
            )
        if widest > self._max_part_half_fp:
            raise ValueError(
                f"part footprint half-extent {widest} m exceeds {self._max_part_half_fp} m: a part seated at the "
                f"edge of the success band (+-{self._belt_b_band_half} m) would hang off belt B's deck"
            )

    _human_render_eye: tuple[float, float, float] = (0.80, 0.00, 1.45)
    _human_render_target: tuple[float, float, float] = (-0.13, 0.00, 0.28)

    @property
    def _default_human_render_camera_configs(self):
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=self._human_render_eye,
            target=self._human_render_target,
        )

    @property
    def _belt_b_center_x(self) -> float:
        return self.workspace_offset[0] + self._belt_b_dx

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        device = self.device

        self.belt_b = build_conveyor(
            self.scene,
            workspace_offset=(self._belt_b_center_x, oy, oz),
            deck_height=self._conveyor_deck_height,
            belt_half_width=self._conveyor_half_width,
            half_length=self._conveyor_half_length,
            static_friction=self.belt_static_friction,
            kinetic_friction=self.belt_kinetic_friction,
            name="conveyor_b",
            texture_slot=1,
        )

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        deck_z = oz + self._conveyor_deck_height

        def _build_rider(names: list[str], label: str) -> tuple[Actor, torch.Tensor, torch.Tensor, torch.Tensor]:
            objs: list[Actor] = []
            for i, name in enumerate(names):
                spec = spec_by_id[name]
                builder = spec.builder_fn(self.scene)
                builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.50])
                builder.set_scene_idxs([i])
                obj = builder.build(name=f"{label}-{name}-{i}")
                self.remove_from_state_dict_registry(obj)
                obj.angular_damping = 5.0
                objs.append(obj)
            merged = Actor.merge(objs, name=label)
            self.add_to_state_dict_registry(merged)
            specs = [spec_by_id[n] for n in names]

            def _tensor(values: list[float]) -> torch.Tensor:
                return torch.as_tensor(values, dtype=torch.float32, device=device)

            return (
                merged,
                _tensor([s.bbox[2] for s in specs]),
                _tensor([max(s.bbox[0], s.bbox[1]) for s in specs]),
                _tensor([s.patch_radius if s.patch_radius is not None else self._part_patch_radius for s in specs]),
            )

        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[n]) for n in self._part_names]
        self.part, self._part_half_z, self._part_half_fp, part_patch = _build_rider(self._part_names, "swap_part")
        specs = [spec_by_id[n] for n in self._part_names]
        self._part_grasp_width = torch.as_tensor(
            [s.grasp_width if s.grasp_width is not None else 2.0 * s.bbox[0] for s in specs],
            dtype=torch.float32,
            device=device,
        )
        self.belt.register_rider(self.part, patch_radius=part_patch)
        self.belt_b.register_rider(self.part, patch_radius=part_patch)
        self.puck = self.part
        self._puck_half_z = self._part_half_z

        self.distractor: Actor | None = None
        if self.randomization.distractor:
            d_names = [str(n) for n in self._batched_episode_rng.choice(np.array(self.objects), replace=True)]
            self.distractor, self._distractor_half_z, self._distractor_half_fp, d_patch = _build_rider(
                d_names, "distractor_rider"
            )
            self.belt.register_rider(self.distractor, patch_radius=d_patch)

        self._release_xy = torch.zeros((self.num_envs, 2), device=device)
        self._release_z_offset = torch.zeros((self.num_envs,), device=device)
        self._part_spawn_up = torch.zeros((self.num_envs, 3), device=device)

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"move the {name} from the far conveyor belt onto the near conveyor belt" for name in self._part_display
        ]

    def _sample_belt_b_velocity(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        sign_a = self.belt.direction_sign[env_idx]
        if not self.randomization.belt_b_speed:
            return -self._belt_b_speed_ratio * self.belt.velocity[env_idx]
        lo, hi = self.randomization.belt_b_speed_range
        speeds = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)
        return -sign_a * speeds

    def _sample_place_y(self, b: int) -> torch.Tensor:
        if not self.randomization.place_y:
            return torch.full((b,), self.workspace_offset[1], device=self.device)
        lo, hi = self.randomization.place_y_range
        offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return self.workspace_offset[1] + torch.as_tensor(offsets, device=self.device)

    def _sample_part_orientation(self, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        if self.randomization.rider_yaw:
            lo, hi = self.randomization.rider_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _spawn_rider(
        self,
        actor: Actor,
        env_idx: torch.Tensor,
        b: int,
        *,
        half_z: torch.Tensor,
        half_fp: torch.Tensor,
        quat: torch.Tensor,
        downstream: float = 0.0,
    ) -> torch.Tensor:
        _ox, _oy, oz = self.workspace_offset
        device = self.device
        pos = torch.zeros((b, 3), device=device)
        pos[:, :2] = self._sample_rider_xy(env_idx, b, half_fp=half_fp, downstream=downstream)
        pos[:, 2] = oz + self._conveyor_deck_height + half_z[env_idx] + 0.001
        actor.set_pose(Pose.create_from_pq(pos, quat))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        actor.set_linear_velocity(vel)
        actor.set_angular_velocity(torch.zeros((b, 3), device=device))
        return pos[:, :2]

    def _sample_rider_xy(
        self, env_idx: torch.Tensor, b: int, *, half_fp: torch.Tensor, downstream: float
    ) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        sign = self.belt.direction_sign[env_idx]
        start_y = self._conveyor_half_length - half_fp[env_idx] - 0.04
        xy = torch.zeros((b, 2), device=self.device)
        xy[:, 0] = ox
        xy[:, 1] = oy + (downstream - start_y) * sign
        return xy

    @property
    def _velocity_belts(self) -> list[ConveyorBelt]:
        return [self.belt, self.belt_b]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self.belt_b.reset(env_idx, self._sample_belt_b_velocity(env_idx, b))

        part_q = self._sample_part_orientation(b)
        self._spawn_rider(self.part, env_idx, b, half_z=self._part_half_z, half_fp=self._part_half_fp, quat=part_q)
        self._part_spawn_up[env_idx] = quaternion_to_matrix(part_q)[:, :, 2]

        if self.distractor is not None:
            d_q = torch.zeros((b, 4), device=device)
            d_q[:, 0] = 1.0
            d_xy = self._spawn_rider(
                self.distractor,
                env_idx,
                b,
                half_z=self._distractor_half_z,
                half_fp=self._distractor_half_fp,
                quat=d_q,
                downstream=self._distractor_pitch,
            )

        self._release_xy[env_idx, 0] = self._belt_b_center_x
        self._release_xy[env_idx, 1] = self._sample_place_y(b)

        self._reset_swap_trackers(env_idx)
        if self.distractor is not None:
            self._distractor_spawn_xy[env_idx] = d_xy
            self._distractor_spawn_q[env_idx] = d_q

    def _before_control_step(self) -> None:
        super()._before_control_step()
        self.belt_b.on_control_step()

    def _before_simulation_step(self) -> None:
        step_belts(self.belt, self.belt_b)

    def _reset_swap_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_swap_sustain"):
            self._swap_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._swap_delivered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._rider_disturbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._distractor_spawn_xy = torch.zeros((self.num_envs, 2), device=self.device)
            self._distractor_spawn_q = torch.zeros((self.num_envs, 4), device=self.device)
            self._distractor_spawn_q[:, 0] = 1.0
            self._gap_rested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._swap_sustain[env_idx] = 0
        self._swap_delivered[env_idx] = False
        self._rider_disturbed[env_idx] = False
        self._gap_rested[env_idx] = False

    def _rider_disturbed_now(self, decoy_pose: Pose, deck_z: float) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        d_pos = decoy_pose.p
        lifted = torch.abs(d_pos[:, 2] - (deck_z + self._distractor_half_z)) > self._rider_z_tol
        shoved = torch.abs(d_pos[:, 0] - ox) > self._rider_x_tol
        on_belt_a = torch.abs(d_pos[:, 1] - oy) <= self._conveyor_half_length - self._distractor_half_fp
        return (lifted | shoved | self._rider_orientation_disturbed(decoy_pose.q)) & on_belt_a

    def _part_settled(self) -> torch.Tensor:
        return (torch.linalg.norm(self.part.angular_velocity, dim=-1) <= self._part_spin_tol) & (
            torch.abs(self.part.linear_velocity[:, 2]) <= self._part_rise_tol
        )

    def _rider_orientation_disturbed(self, decoy_q: torch.Tensor) -> torch.Tensor:
        return quaternion_to_matrix(decoy_q)[:, 2, 2] < math.cos(math.radians(self._rider_tilt_deg))

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        bx = self._belt_b_center_x
        part_pose = self.part.pose
        part_pos = part_pose.p
        part_vel = self.part.linear_velocity
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        belt_b_v = self.belt_b.velocity

        on_deck = torch.abs(part_pos[:, 2] - (deck_z + self._part_half_z)) <= self._deck_rest_tol
        in_band = torch.abs(part_pos[:, 0] - bx) <= self._belt_b_band_half
        seated = on_deck & in_band

        part_up = quaternion_to_matrix(part_pose.q)[:, :, 2]
        not_tipped = (part_up * self._part_spawn_up).sum(dim=-1) >= math.cos(math.radians(self._tip_max_deg))

        comoving = (
            (torch.abs(part_vel[:, 1] - belt_b_v) <= self._comove_tol)
            & (torch.abs(part_vel[:, 0]) <= self._lateral_speed_tol)
            & self._part_settled()
        )

        finger_width = self.gripper_aperture()
        fingers_open = finger_width >= self._part_grasp_width + self._release_finger_margin
        hand_away = torch.linalg.norm(tcp_pos - part_pos, dim=-1) > self._released_distance
        released = fingers_open & hand_away

        tcp_clear = (tcp_pos[:, 2] - deck_z) >= self._tcp_clear_above_deck

        on_belt_b_run = torch.abs(part_pos[:, 1] - oy) <= self._belt_b_run_y
        belt_b_travelling = part_vel[:, 1] * belt_b_v >= 0.5 * belt_b_v * belt_b_v
        on_belt_b = on_belt_b_run & belt_b_travelling

        swapping = seated & not_tipped & comoving & released & tcp_clear & on_belt_b
        self._swap_sustain = self._advance_sustain("belt_swap_deliver", self._swap_sustain, swapping)

        along_slip = torch.zeros_like(belt_b_v)
        if self.distractor is not None:
            self._rider_disturbed = self._rider_disturbed | (
                self._rider_disturbed_now(self.distractor.pose, deck_z) & ~self._swap_delivered
            )
            along_slip = torch.abs(self.distractor.linear_velocity[:, 1] - self.belt.velocity)
        self._swap_delivered = self._swap_delivered | (self._swap_sustain >= self._swap_sustain_steps)

        underside = part_pos[:, 2] - self._part_half_z
        on_surface = torch.abs(underside - oz) < self._gap_rest_tol
        clear_of_a = torch.abs(part_pos[:, 0] - ox) > self._conveyor_half_width + self._part_half_fp
        clear_of_b = torch.abs(part_pos[:, 0] - bx) > self._conveyor_half_width + self._part_half_fp
        self._gap_rested = self._gap_rested | (on_surface & clear_of_a & clear_of_b)

        success = (self._swap_sustain >= self._swap_sustain_steps) & ~self._rider_disturbed & ~self._gap_rested

        return {
            "success": success,
            "seated": seated,
            "on_deck": on_deck,
            "in_band": in_band,
            "not_tipped": not_tipped,
            "comoving": comoving,
            "released": released,
            "fingers_open": fingers_open,
            "hand_away": hand_away,
            "tcp_clear": tcp_clear,
            "on_belt_b": on_belt_b,
            "on_belt_b_run": on_belt_b_run,
            "belt_b_travelling": belt_b_travelling,
            "swapping": swapping,
            "swap_sustain": self._swap_sustain,
            "delivered": self._swap_delivered,
            "rider_disturbed": self._rider_disturbed,
            "gap_rested": self._gap_rested,
            "distractor_along_slip": along_slip,
            "part_to_release_xy": torch.linalg.norm(part_pos[:, :2] - self._release_xy, dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        _ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - part_pos, dim=-1))
        to_belt_b = torch.stack(
            [part_pos[:, 0] - self._belt_b_center_x, part_pos[:, 2] - (deck_z + self._part_half_z)],
            dim=-1,
        )
        transfer_reward = 1 - torch.tanh(3 * torch.linalg.norm(to_belt_b, dim=-1))
        riding = (info["seated"] & info["comoving"]).float()
        clean = (~info["rider_disturbed"] & ~info["gap_rested"]).float()
        return reach_reward + transfer_reward + 2.0 * riding * clean

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
