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
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_key_slot.assets import build_keyed_carrier, build_studded_part
from sweep_bench.tasks.conveyor_key_slot.config import RandomizationConfig
from sweep_bench.tasks.conveyor_key_slot.scripted import ScriptedKeySlotMixin
from sweep_bench.tasks.conveyor_key_slot.variants import (
    DECOY_COLOR,
    DECOY_HALF_Y,
    KEY_SLOT_OBJECT_IDS,
    KEY_SLOT_PART_SPECS,
    PART_BODY_FRICTION,
    PART_DENSITY,
    PART_HALF_X,
    PART_HALF_Z,
    PART_HEAD_HALF,
    PART_HEAD_HALF_THICKNESS,
    PART_KEY_FRICTION,
    PART_STUD_LENGTH,
    PART_STUD_RADIUS,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorKeySlot-v0")
@register_env("ConveyorKeySlot-v0", max_episode_steps=300)
class ConveyorKeySlotEnv(ScriptedKeySlotMixin, ConveyorWorkspaceEnv):
    """Free a keyed part from its slot on a belt-riding carrier: slide, then lift."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(KEY_SLOT_OBJECT_IDS, extra=KEY_SLOT_PART_SPECS)

    _carrier_half_x: float = 0.060
    _carrier_half_y: float = 0.080
    _base_half_thickness: float = 0.004
    _cavity_height: float = 0.012
    _plate_thickness: float = 0.006
    _rail_thickness: float = 0.012
    _carrier_density: float = 1800.0
    _carrier_patch_radius: float = 0.040
    _plate_friction: float = 0.08

    _carrier_color: tuple[float, float, float, float] = (0.35, 0.36, 0.40, 1.0)
    _plate_color: tuple[float, float, float, float] = (0.62, 0.64, 0.68, 1.0)

    _keyhole_ys: tuple[float, float] = (0.028, -0.028)
    _open_x: float = -0.020
    _channel_width: float = 0.013
    _channel_end_clear: float = 0.001
    _chamfer: float = 0.005

    _disengage_height: float = 0.010
    _grip_gap_tol: float = 0.010
    _hold_dist: float = 0.05
    _decoy_capture_margin: float = 0.002
    _carrier_lateral_tol: float = 0.05
    _carrier_y_limit: float = 0.85
    _carrier_max_tilt_cos: float = math.cos(math.radians(10.0))
    _carrier_speed_tol_base: float = 0.06
    _carrier_speed_tol_per_speed: float = 0.20
    _key_sustain_steps: int = 6

    default_prompt: str = "slide the keyed part along its slot and lift it off the carrier on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        channel_len: float = 0.030,
        hole_size: float = 0.032,
        **kwargs,
    ) -> None:
        min_hole = 2 * PART_HEAD_HALF + 2 * self._min_head_clearance
        if hole_size < min_hole:
            raise ValueError(
                f"hole_size {hole_size} is below the head-clearance floor {min_hole:.4f} "
                f"(head {2 * PART_HEAD_HALF} + {self._min_head_clearance} of centring slack per side): "
                "the head could not pass the opening at all"
            )
        if self._channel_width >= 2 * PART_HEAD_HALF:
            raise ValueError(
                f"channel width {self._channel_width} does not block the {2 * PART_HEAD_HALF} m head: "
                "a straight lift would not be geometrically blocked"
            )
        if channel_len < 0.5 * hole_size + PART_HEAD_HALF:
            raise ValueError(
                f"channel_len {channel_len} is shorter than {0.5 * hole_size + PART_HEAD_HALF:.4f}: "
                "the seated head would already overlap the square opening"
            )
        self.channel_len: float = channel_len
        self.hole_size: float = hole_size
        super().__init__(*args, **kwargs)

    _min_head_clearance: float = 0.004

    @property
    def _base_top_z(self) -> float:
        return 2.0 * self._base_half_thickness

    @property
    def _plate_bottom_z(self) -> float:
        return self._base_top_z + self._cavity_height

    @property
    def _plate_top_z(self) -> float:
        return self._plate_bottom_z + self._plate_thickness

    @property
    def _open_half(self) -> float:
        return 0.5 * self.hole_size

    @property
    def _seated_x(self) -> float:
        return self._open_x + self.channel_len

    @property
    def _slide_len(self) -> float:
        return self.channel_len

    @property
    def _channel_x_hi(self) -> float:
        return self._seated_x + PART_STUD_RADIUS + self._channel_end_clear

    @property
    def _main_keyhole_y(self) -> float:
        return self._keyhole_ys[0]

    @property
    def _part_origin_z(self) -> float:
        return self._plate_top_z + PART_HALF_Z

    @property
    def _head_local_dz(self) -> float:
        return -(PART_HALF_Z + PART_STUD_LENGTH + PART_HEAD_HALF_THICKNESS)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_keyed_carrier(
            self.scene,
            half_x=self._carrier_half_x,
            half_y=self._carrier_half_y,
            base_half_thickness=self._base_half_thickness,
            cavity_height=self._cavity_height,
            plate_thickness=self._plate_thickness,
            rail_thickness=self._rail_thickness,
            keyhole_ys=sorted(self._keyhole_ys),
            open_x=self._open_x,
            open_half=self._open_half,
            channel_half_w=0.5 * self._channel_width,
            channel_x_hi=self._channel_x_hi,
            chamfer=self._chamfer,
            density=self._carrier_density,
            body_color=self._carrier_color,
            plate_color=self._plate_color,
            plate_friction=self._plate_friction,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.carrier = carrier_builder.build(name="key_slot_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        drawn = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in drawn]
        self._part_display: list[str] = [display_name(spec_by_id[n]) for n in self._part_names]

        parts: list[Actor] = []
        for i, name in enumerate(self._part_names):
            builder = spec_by_id[name].builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._part_origin_z])
            builder.set_scene_idxs([i])
            part = builder.build(name=f"keyed_part-{name}-{i}")
            self.remove_from_state_dict_registry(part)
            parts.append(part)
        self.part = Actor.merge(parts, name="keyed_part")
        self.add_to_state_dict_registry(self.part)

        decoy_builder = build_studded_part(
            self.scene,
            body_half_x=PART_HALF_X,
            body_half_y=DECOY_HALF_Y,
            body_half_z=PART_HALF_Z,
            stud_radius=PART_STUD_RADIUS,
            stud_length=PART_STUD_LENGTH,
            head_half=PART_HEAD_HALF,
            head_half_thickness=PART_HEAD_HALF_THICKNESS,
            color=DECOY_COLOR,
            density=PART_DENSITY,
            body_friction=PART_BODY_FRICTION,
            key_friction=PART_KEY_FRICTION,
            initial_pose=sapien.Pose(p=[ox, oy, deck_z + self._part_origin_z]),
        )
        self.decoy = decoy_builder.build(name="decoy_part")

        widths = [spec_by_id[n].grasp_width for n in self._part_names]
        self._part_grip_width = torch.as_tensor(
            [w if w is not None else 2 * PART_HALF_X for w in widths], dtype=torch.float32, device=self.device
        )
        head_offset = torch.zeros((self.num_envs, 3), device=self.device)
        head_offset[:, 2] = self._head_local_dz
        self._head_offset_pose = Pose.create_from_pq(p=head_offset)

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"slide the {name} along its slot and lift it off the carrier" for name in self._part_display]

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.part, self.decoy]

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
        for actor, keyhole_y in ((self.part, self._keyhole_ys[0]), (self.decoy, self._keyhole_ys[1])):
            local = torch.zeros((b, 3), device=device)
            local[:, 0] = self._seated_x
            local[:, 1] = keyhole_y
            local[:, 2] = self._part_origin_z
            actor.set_pose(
                Pose.create_from_pq(carrier_pos + torch.bmm(rot, local.unsqueeze(-1)).squeeze(-1), carrier_q)
            )
            actor.set_linear_velocity(vel)
            actor.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_key_trackers(env_idx)

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

    def _carrier_rotation(self) -> torch.Tensor:
        return quaternion_to_matrix(self.carrier.pose.q)

    def _carrier_local(self, world_points: torch.Tensor) -> torch.Tensor:
        rel = world_points - self.carrier.pose.p
        return torch.bmm(self._carrier_rotation().transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)

    def _head_local(self, actor: Actor) -> torch.Tensor:
        return self._carrier_local((actor.pose * self._head_offset_pose).p)

    def _part_head_local(self) -> torch.Tensor:
        return self._head_local(self.part)

    def _part_is_held(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        separation = self.gripper_aperture()
        pinched = torch.abs(separation - self._part_grip_width) < self._grip_gap_tol
        at_hand = torch.linalg.norm(self.part.pose.p - agent.tcp.pose.p, dim=-1) <= self._hold_dist
        return pinched & at_hand

    def _reset_key_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_key_sustain_counter"):
            self._key_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._key_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, _oz = self.workspace_offset
        carrier_pos = self.carrier.pose.p

        head = self._head_local(self.part)
        decoy_head = self._head_local(self.decoy)

        head_high = head[:, 2] >= self._plate_top_z + self._disengage_height
        on_plate = (torch.abs(head[:, 0]) <= self._carrier_half_x) & (torch.abs(head[:, 1]) <= self._carrier_half_y)
        disengaged = head_high & on_plate

        part_held = self._part_is_held()

        decoy_captive = decoy_head[:, 2] <= self._plate_bottom_z - self._decoy_capture_margin

        on_belt = (torch.abs(carrier_pos[:, 0] - ox) <= self._carrier_lateral_tol) & (
            torch.abs(carrier_pos[:, 1] - oy) < self._carrier_y_limit
        )

        belt_v = self.belt.velocity
        speed_tol = self._carrier_speed_tol_base + self._carrier_speed_tol_per_speed * tolerance_speed(belt_v)
        tracking = torch.abs(self.carrier.linear_velocity[:, 1] - belt_v) <= speed_tol
        upright = self._carrier_rotation()[:, 2, 2] > self._carrier_max_tilt_cos
        carrier_riding = on_belt & tracking & upright

        holding = disengaged & part_held & decoy_captive & carrier_riding
        self._key_sustain_counter = self._advance_sustain("key_slot_sustain", self._key_sustain_counter, holding)
        success = self._key_sustain_counter >= self._key_sustain_steps

        return {
            "success": success,
            "head_height": head[:, 2],
            "head_high": head_high,
            "head_on_plate": on_plate,
            "disengaged": disengaged,
            "part_held": part_held,
            "decoy_head_height": decoy_head[:, 2],
            "decoy_captive": decoy_captive,
            "carrier_on_belt": on_belt,
            "carrier_tracking": tracking,
            "carrier_upright": upright,
            "carrier_riding": carrier_riding,
            "key_sustain": self._key_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_part = torch.linalg.norm(tcp_pos - self.part.pose.p, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_part)

        head_x = self._head_local(self.part)[:, 0]
        slide_reward = ((self._seated_x - head_x) / self._slide_len).clamp(0.0, 1.0)

        seated_head_z = self._part_origin_z + self._head_local_dz
        target = self._plate_top_z + self._disengage_height
        lift_reward = ((self._head_local(self.part)[:, 2] - seated_head_z) / (target - seated_head_z)).clamp(0.0, 1.0)

        held = info["part_held"].float()
        return reach_reward + 2.0 * slide_reward * held + 2.0 * lift_reward * held

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
