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
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed
from sweep_bench.tasks.conveyor_rail_balance.assets import build_rail_carrier
from sweep_bench.tasks.conveyor_rail_balance.config import RandomizationConfig
from sweep_bench.tasks.conveyor_rail_balance.scripted import ScriptedRailBalanceMixin
from sweep_bench.tasks.conveyor_rail_balance.variants import (
    BALANCE_BLOCK_SPEC,
    RAIL_BALANCE_OBJECT_IDS,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec

_FIT_EPS = 1e-6


@bench_task("ConveyorRailBalance-v0")
@register_env("ConveyorRailBalance-v0", max_episode_steps=300)
class ConveyorRailBalanceEnv(ScriptedRailBalanceMixin, ConveyorWorkspaceEnv):
    """Balance a held block on a narrow rail riding the belt, hands off."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(RAIL_BALANCE_OBJECT_IDS, extra=(BALANCE_BLOCK_SPEC,))

    _carrier_half_x: float = 0.055
    _carrier_half_y: float = 0.070
    _carrier_plate_half_thickness: float = 0.006
    _rail_half_width: float = 0.010
    _rail_half_length: float = 0.060
    _rail_height: float = 0.035

    _carrier_patch_radius: float = 0.040
    _block_patch_radius: float = 0.025

    _block_min_footprint: float = 0.045
    _block_max_grasp_width: float = 0.070
    _block_required_tag: str = "box"

    _on_rail_z_tol: float = 0.004
    _balance_band_half: float = 0.008
    _block_tilt_cos: float = math.cos(math.radians(6.0))
    _riding_rel_speed: float = 0.03
    _hands_off_finger_margin: float = 0.01
    _hands_off_distance: float = 0.09
    _carrier_tilt_cos: float = math.cos(math.radians(10.0))
    _travel_fraction: float = 0.5
    _travel_speed_slack: float = 0.012
    _balance_sustain_steps: int = 12

    default_prompt: str = "balance the held block on the rail on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _carrier_origin_above_deck(self) -> float:
        return 2.0 * self._carrier_plate_half_thickness

    def _check_block_fits(self, spec: ObjectSpec) -> None:
        if self._block_required_tag not in spec.tags:
            raise ValueError(
                f"block {spec.id!r} is not tagged {self._block_required_tag!r}; the ON-RAIL and NOT-TIPPED terms "
                f"both measure a flat face resting on the rail's top plane, so only flat-faced blocks balance "
                f"rather than rock (see variants.py)"
            )
        if abs(spec.bbox[0] - spec.bbox[1]) > _FIT_EPS:
            raise ValueError(
                f"block {spec.id!r} has a rectangular footprint {spec.bbox[:2]}; the gripper closes on the "
                f"cross-belt axis, which is also the axis that overhangs the rail, so a non-square block's "
                f"held width and balanced width would disagree (see variants.py)"
            )
        footprint = 2.0 * spec.bbox[0]
        if footprint < self._block_min_footprint - _FIT_EPS:
            raise ValueError(
                f"block {spec.id!r} is only {footprint} m across; below {self._block_min_footprint} m it barely "
                f"overhangs the {2.0 * self._rail_half_width} m rail and the balance stops being under-supported"
            )
        grasp = spec.grasp_width if spec.grasp_width is not None else footprint
        if grasp > self._block_max_grasp_width + _FIT_EPS:
            raise ValueError(
                f"block {spec.id!r} needs a {grasp} m grip; past {self._block_max_grasp_width} m the panda cannot "
                f"open the extra {self._hands_off_finger_margin} m the HANDS OFF term requires"
            )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_rail_carrier(
            self.scene,
            plate_half_x=self._carrier_half_x,
            plate_half_y=self._carrier_half_y,
            plate_half_thickness=self._carrier_plate_half_thickness,
            rail_half_width=self._rail_half_width,
            rail_half_length=self._rail_half_length,
            rail_height=self._rail_height,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._carrier_origin_above_deck])
        self.carrier = carrier_builder.build(name="rail_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._block_names: list[str] = [str(name) for name in names]
        self._block_display: list[str] = [display_name(spec_by_id[name]) for name in self._block_names]
        specs = [spec_by_id[name] for name in self._block_names]
        for spec in specs:
            self._check_block_fits(spec)

        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.50])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"block-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        self.block = Actor.merge(objs, name="block")
        self.add_to_state_dict_registry(self.block)
        self.belt.register_rider(
            self.block,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._block_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self._block_half_z = torch.as_tensor([s.bbox[2] for s in specs], dtype=torch.float32, device=self.device)
        self._block_grasp_width = torch.as_tensor(
            [s.grasp_width if s.grasp_width is not None else 2.0 * s.bbox[0] for s in specs],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def episode_objects(self) -> list[str]:
        return list(self._block_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"balance the {name} on the rail on the conveyor belt" for name in self._block_display]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = 0.5 * self._block_grasp_width[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_pos = self._sample_carrier_position(env_idx, b)
        carrier_q = torch.zeros((b, 4), device=device)
        carrier_q[:, 0] = 1.0
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
        block_q = torch.zeros((b, 4), device=device)
        block_q[:, 0] = 1.0
        self.block.set_pose(Pose.create_from_pq(tcp_pos, block_q))
        self.block.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.block.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_balance_trackers(env_idx)

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.carrier_lateral_offset:
            lo, hi = self.randomization.carrier_x_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._carrier_origin_above_deck
        return pos

    def _reset_balance_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_balance_sustain_counter"):
            self._balance_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._balance_sustain_counter[env_idx] = 0

    def evaluate(self):
        agent = cast("Panda", self.agent)
        carrier_pose = self.carrier.pose
        carrier_pos = carrier_pose.p
        block_pose = self.block.pose
        block_pos = block_pose.p
        tcp_pos = agent.tcp.pose.p

        local = (carrier_pose.inv() * Pose.create_from_pq(block_pos)).p

        bottom_gap = torch.abs((local[:, 2] - self._block_half_z) - self._rail_height)
        on_rail = bottom_gap <= self._on_rail_z_tol

        rail_offset = torch.abs(local[:, 0])
        balanced = rail_offset < self._balance_band_half

        up_local = torch.zeros((self.num_envs, 3), device=self.device)
        up_local[:, 2] = 1.0
        carrier_up = (carrier_pose * Pose.create_from_pq(up_local)).p - carrier_pos
        block_up = (block_pose * Pose.create_from_pq(up_local)).p - block_pos
        upright_cos = (carrier_up * block_up).sum(dim=-1)
        not_tipped = upright_cos >= self._block_tilt_cos

        finger_width = self.gripper_aperture()
        hands_off = (finger_width > self._block_grasp_width + self._hands_off_finger_margin) & (
            torch.linalg.norm(tcp_pos - block_pos, dim=-1) > self._hands_off_distance
        )

        rel_speed = torch.linalg.norm(self.block.linear_velocity - self.carrier.linear_velocity, dim=-1)
        riding = rel_speed < self._riding_rel_speed

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        tilt_cos = carrier_up[:, 2]
        sign = self.belt.direction_sign
        ground_speed = self.carrier.linear_velocity[:, 1] * sign
        min_ground_speed = carried_speed(torch.abs(self.belt.velocity), self._travel_fraction)
        carrier_on_belt = (
            (carrier_pos[:, 2] > deck_z)
            & (torch.abs(carrier_pos[:, 0] - ox) < self._conveyor_half_width + self._carrier_half_x)
            & (torch.abs(carrier_pos[:, 1] - oy) < self._conveyor_half_length + self._carrier_half_y)
            & (tilt_cos >= self._carrier_tilt_cos)
            & (ground_speed + self._travel_speed_slack >= min_ground_speed)
        )

        balancing = on_rail & balanced & not_tipped & hands_off & riding & carrier_on_belt

        self._balance_sustain_counter = self._advance_sustain(
            "balance_sustain", self._balance_sustain_counter, balancing
        )
        success = self._balance_sustain_counter >= self._balance_sustain_steps

        return {
            "success": success,
            "on_rail": on_rail,
            "balanced": balanced,
            "not_tipped": not_tipped,
            "hands_off": hands_off,
            "riding": riding,
            "carrier_on_belt": carrier_on_belt,
            "rail_offset": rail_offset,
            "balance_sustain": self._balance_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        block_pos = self.block.pose.p
        carrier_pos = self.carrier.pose.p

        xy_dist = torch.linalg.norm(block_pos[:, :2] - carrier_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        band_reward = (1.0 - info["rail_offset"] / self._balance_band_half).clamp(0.0, 1.0)

        return carry_reward + 2.0 * band_reward * info["on_rail"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
