from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_rel_place.assets import build_anchor_block, build_long_tray
from sweep_bench.tasks.conveyor_rel_place.config import RandomizationConfig
from sweep_bench.tasks.conveyor_rel_place.scripted import ScriptedRelPlaceMixin
from sweep_bench.tasks.conveyor_rel_place.variants import REL_PLACE_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec

_TRAY_COLOR = (0.34, 0.38, 0.46, 1.0)
_TRAY_PAD_COLOR = (0.18, 0.18, 0.20, 1.0)
_ANCHOR_COLOR = (0.90, 0.45, 0.10, 1.0)

_RELATION_PHRASE: dict[int, str] = {1: "in front of", -1: "behind"}


@bench_task("ConveyorRelPlace-v0")
@register_env("ConveyorRelPlace-v0", max_episode_steps=300)
class ConveyorRelPlaceEnv(ScriptedRelPlaceMixin, ConveyorWorkspaceEnv):
    """Set the held block down IN FRONT OF or BEHIND an anchor on a moving tray."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(REL_PLACE_OBJECT_IDS)

    _tray_half_x: float = 0.065
    _tray_half_y: float = 0.18
    _floor_half_height: float = 0.005
    _floor_pad_half_height: float = 0.0015
    _floor_pad_static_friction: float = 0.9
    _floor_pad_dynamic_friction: float = 0.8
    _lip_half_thickness: float = 0.003
    _lip_half_height: float = 0.0075

    _anchor_half_x: float = 0.025
    _anchor_half_y: float = 0.025
    _anchor_half_z: float = 0.025
    _anchor_density: float = 1000.0
    _anchor_static_friction: float = 0.9
    _anchor_dynamic_friction: float = 0.8

    _tray_patch_radius: float = 0.055
    _block_patch_radius: float = 0.015

    _zone_near: float = 0.05
    _zone_far: float = 0.15
    _zone_half_x: float = 0.030
    _floor_contact_tol: float = 0.006
    _comove_tol: float = 0.05
    _released_finger_margin: float = 0.01
    _released_distance_threshold: float = 0.09
    _anchor_drift_tol: float = 0.02
    _tray_x_tol: float = 0.04
    _place_sustain_steps: int = 10

    _default_relation_sign: int = 1

    default_prompt: str = "place the block in front of the marker block on the tray on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _tray_inner_half_x(self) -> float:
        return self._tray_half_x - 2.0 * self._lip_half_thickness

    @property
    def _tray_inner_half_y(self) -> float:
        return self._tray_half_y - 2.0 * self._lip_half_thickness

    @property
    def _zone_center(self) -> float:
        return 0.5 * (self._zone_near + self._zone_far)

    @property
    def _anchor_rest_local(self) -> torch.Tensor:
        rest = torch.zeros(3, device=self.device)
        rest[2] = self._floor_half_height + self._anchor_half_z
        return rest

    def _to_world(self, tray_pose: Pose, local: torch.Tensor) -> torch.Tensor:
        return (tray_pose * Pose.create_from_pq(local)).p

    def _in_tray_frame(self, world: torch.Tensor) -> torch.Tensor:
        return (self.tray.pose.inv() * Pose.create_from_pq(world)).p

    def relation_phrases(self) -> list[str]:
        return [_RELATION_PHRASE[int(s)] for s in self._relation_sign.cpu().tolist()]

    def target_zone_offset_local(self) -> torch.Tensor:
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 1] = self._relation_sign * self.belt.direction_sign * self._zone_center
        return offset

    def target_zone_world_pos(self) -> torch.Tensor:
        return self._to_world(self.tray.pose, self.target_zone_offset_local())

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        tray_builder = build_long_tray(
            self.scene,
            half_x=self._tray_half_x,
            half_y=self._tray_half_y,
            floor_half_height=self._floor_half_height,
            pad_half_height=self._floor_pad_half_height,
            lip_half_thickness=self._lip_half_thickness,
            lip_half_height=self._lip_half_height,
            color=_TRAY_COLOR,
            pad_color=_TRAY_PAD_COLOR,
            pad_static_friction=self._floor_pad_static_friction,
            pad_dynamic_friction=self._floor_pad_dynamic_friction,
        )
        tray_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._floor_half_height + 0.001])
        self.tray = tray_builder.build(name="rel_place_tray")
        self.belt.register_rider(self.tray, patch_radius=self._tray_patch_radius)

        anchor_builder = build_anchor_block(
            self.scene,
            half_x=self._anchor_half_x,
            half_y=self._anchor_half_y,
            half_z=self._anchor_half_z,
            density=self._anchor_density,
            color=_ANCHOR_COLOR,
            static_friction=self._anchor_static_friction,
            dynamic_friction=self._anchor_dynamic_friction,
        )
        anchor_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.40])
        self.anchor = anchor_builder.build(name="rel_place_anchor")

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._block_names: list[str] = [str(name) for name in names]
        self._block_display: list[str] = [display_name(spec_by_id[name]) for name in self._block_names]
        specs = [spec_by_id[name] for name in self._block_names]
        self.block = self._build_block_group(specs, "rel_place_block", deck_z + 0.50)
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
            [s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1]) for s in specs],
            dtype=torch.float32,
            device=self.device,
        )

    def _build_block_group(self, specs: list[ObjectSpec], group: str, park_z: float) -> Actor:
        ox, oy, _oz = self.workspace_offset
        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, park_z])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{group}-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        merged = Actor.merge(objs, name=group)
        self.add_to_state_dict_registry(merged)
        return merged

    @property
    def episode_objects(self) -> list[str]:
        return list(self._block_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"place the {name} {phrase} the marker block on the tray on the conveyor belt"
            for name, phrase in zip(self._block_display, self.relation_phrases(), strict=True)
        ]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = 0.5 * self._block_grasp_width[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_place_trackers(env_idx)
        self._relation_sign[env_idx] = self._sample_relation(b)

        tray_pos = self._sample_tray_position(env_idx, b)
        tray_q = torch.zeros((b, 4), device=device)
        tray_q[:, 0] = 1.0
        tray_pose = Pose.create_from_pq(tray_pos, tray_q)
        self.tray.set_pose(tray_pose)
        tray_vel = torch.zeros((b, 3), device=device)
        tray_vel[:, 1] = self.belt.velocity[env_idx]
        self.tray.set_linear_velocity(tray_vel)
        self.tray.set_angular_velocity(torch.zeros((b, 3), device=device))

        anchor_local = self._anchor_rest_local.expand(b, 3).clone()
        anchor_local[:, 2] += 0.0005
        self.anchor.set_pose(Pose.create_from_pq(self._to_world(tray_pose, anchor_local), tray_q))
        self.anchor.set_linear_velocity(tray_vel)
        self.anchor.set_angular_velocity(torch.zeros((b, 3), device=device))

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

    def _sample_relation(self, b: int) -> torch.Tensor:
        if self.randomization.relation:
            picks = self._episode_rng.choice(np.array([1.0, -1.0], dtype=np.float32), size=b)
            return torch.as_tensor(picks, dtype=torch.float32, device=self.device)
        return torch.full((b,), float(self._default_relation_sign), dtype=torch.float32, device=self.device)

    def _sample_tray_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._tray_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._floor_half_height + 0.001
        return pos

    def _reset_place_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_place_sustain_counter"):
            self._place_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._relation_sign = torch.full(
                (self.num_envs,), float(self._default_relation_sign), dtype=torch.float32, device=self.device
            )
            self._anchor_disturbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._place_sustain_counter[env_idx] = 0
        self._anchor_disturbed[env_idx] = False

    def evaluate(self):
        agent = cast("Panda", self.agent)
        block_pos = self.block.pose.p
        belt_v = self.belt.velocity

        block_local = self._in_tray_frame(block_pos)

        along = block_local[:, 1] * self.belt.direction_sign * self._relation_sign
        on_commanded_side = (along >= self._zone_near) & (along <= self._zone_far)

        centred_in_x = torch.abs(block_local[:, 0]) < self._zone_half_x

        block_bottom_local = block_local[:, 2] - self._block_half_z
        on_tray_floor = torch.abs(block_bottom_local - self._floor_half_height) <= self._floor_contact_tol

        riding = torch.linalg.norm(self.block.linear_velocity - self.tray.linear_velocity, dim=-1) < self._comove_tol

        finger_width = self.gripper_aperture()
        released = (finger_width > self._block_grasp_width + self._released_finger_margin) & (
            torch.linalg.norm(agent.tcp.pose.p - block_pos, dim=-1) > self._released_distance_threshold
        )

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        tray_on_belt = rider_on_belt(
            pose=self.tray.pose,
            linear_velocity=self.tray.linear_velocity,
            belt_velocity=belt_v,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._tray_x_tol,
            y_tol=self._conveyor_half_length + self._tray_half_y,
            rest_height=self._floor_half_height,
        )

        anchor_drift = torch.linalg.norm(self._in_tray_frame(self.anchor.pose.p) - self._anchor_rest_local, dim=-1)
        self._anchor_disturbed = self._anchor_disturbed | (anchor_drift > self._anchor_drift_tol)
        anchor_intact = ~self._anchor_disturbed

        placed = on_commanded_side & centred_in_x & on_tray_floor & riding & released & tray_on_belt & anchor_intact

        self._place_sustain_counter = self._advance_sustain("place_sustain", self._place_sustain_counter, placed)
        success = self._place_sustain_counter >= self._place_sustain_steps

        return {
            "success": success,
            "on_commanded_side": on_commanded_side,
            "centred_in_x": centred_in_x,
            "on_tray_floor": on_tray_floor,
            "riding": riding,
            "released": released,
            "tray_on_belt": tray_on_belt,
            "anchor_intact": anchor_intact,
            "anchor_drift": anchor_drift,
            "place_sustain": self._place_sustain_counter,
            "block_zone_offset": along,
        }

    def compute_dense_reward(self, obs, action, info):
        block_pos = self.block.pose.p
        zone_pos = self.target_zone_world_pos()

        xy_dist = torch.linalg.norm(block_pos[:, :2] - zone_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        seated = (info["on_commanded_side"] & info["on_tray_floor"] & info["anchor_intact"]).float()
        return carry_reward + 2.0 * seated

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
