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
from sweep_bench.tasks.conveyor_clear_arch.assets import build_arch_pallet
from sweep_bench.tasks.conveyor_clear_arch.config import RandomizationConfig
from sweep_bench.tasks.conveyor_clear_arch.scripted import ScriptedClearArchMixin
from sweep_bench.tasks.conveyor_clear_arch.variants import CLEAR_ARCH_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec


@bench_task("ConveyorClearArch-v0")
@register_env("ConveyorClearArch-v0", max_episode_steps=300)
class ConveyorClearArchEnv(ScriptedClearArchMixin, ConveyorWorkspaceEnv):
    """Load the pallet bin, then get the hand out before the pallet's arch sweeps through."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(CLEAR_ARCH_OBJECT_IDS)

    _bin_inner_half: float = 0.052
    _bin_wall_half_thickness: float = 0.005
    _bin_wall_half_height: float = 0.018
    _bin_chamfer: float = 0.005
    _plate_half_thickness: float = 0.004
    _plate_half_x: float = 0.062
    _arch_offset: float = 0.25
    _arch_clear_height: float = 0.11
    _rail_half_x: float = 0.062
    _rail_half_y: float = 0.008
    _rail_half_z: float = 0.008
    _upright_half_x: float = 0.007
    _upright_half_y: float = 0.008
    _upright_local_x: float = 0.054
    _plate_back_margin: float = 0.020

    _pallet_patch_radius: float = 0.05
    _part_patch_radius: float = 0.02

    _containment_margin: float = 0.005
    _below_rim_depth: float = 0.020
    _released_finger_width: float = 0.060
    _comove_tol: float = 0.06
    _pallet_comove_tol: float = 0.08
    _pallet_x_tol: float = 0.05
    _pallet_tilt_cos: float = math.cos(math.radians(10.0))

    _slab_half_y: float = 0.045
    _slab_half_x: float = 0.070
    _slab_top_margin: float = 0.010
    _deadline_clear_dist: float = 0.05
    _clear_sustain_steps: int = 6

    default_prompt: str = "put the part in the pallet bin and pull your hand clear of the arch"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        arch_offset: float | None = None,
        arch_clear_height: float | None = None,
        **kwargs,
    ) -> None:
        if arch_offset is not None:
            self._arch_offset = float(arch_offset)
        if arch_clear_height is not None:
            self._arch_clear_height = float(arch_clear_height)
        super().__init__(*args, **kwargs)

    @property
    def _bin_outer_half(self) -> float:
        return self._bin_inner_half + 2.0 * self._bin_wall_half_thickness

    @property
    def _plate_front_y(self) -> float:
        return self._bin_outer_half

    @property
    def _plate_back_y(self) -> float:
        return -(self._arch_offset + self._rail_half_y + self._plate_back_margin)

    @property
    def _bin_floor_local_z(self) -> float:
        return self._plate_half_thickness

    @property
    def _bin_rim_local_z(self) -> float:
        return self._plate_half_thickness + 2.0 * self._bin_wall_half_height

    @property
    def _bin_mouth_local_z(self) -> float:
        return self._bin_rim_local_z + self._bin_chamfer

    @property
    def _rail_top_local_z(self) -> float:
        return self._plate_half_thickness + self._arch_clear_height + 2.0 * self._rail_half_z

    def _arch_local_offset(self, b: int) -> torch.Tensor:
        offset = torch.zeros((b, 3), device=self.device)
        offset[:, 1] = -self._arch_offset
        offset[:, 2] = self._plate_half_thickness + self._arch_clear_height + self._rail_half_z
        return offset

    def _to_world(self, pallet_pose: Pose, local: torch.Tensor) -> torch.Tensor:
        return (pallet_pose * Pose.create_from_pq(local)).p

    def _in_pallet_frame(self, world: torch.Tensor) -> torch.Tensor:
        return (self.pallet.pose.inv() * Pose.create_from_pq(world)).p

    def arch_world_pos(self) -> torch.Tensor:
        return self._to_world(self.pallet.pose, self._arch_local_offset(self.num_envs))

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        pallet_builder = build_arch_pallet(
            self.scene,
            bin_inner_half=self._bin_inner_half,
            bin_wall_half_thickness=self._bin_wall_half_thickness,
            bin_wall_half_height=self._bin_wall_half_height,
            plate_half_thickness=self._plate_half_thickness,
            plate_half_x=self._plate_half_x,
            plate_back_y=self._plate_back_y,
            plate_front_y=self._plate_front_y,
            arch_offset=self._arch_offset,
            arch_clear_height=self._arch_clear_height,
            rail_half_x=self._rail_half_x,
            rail_half_y=self._rail_half_y,
            rail_half_z=self._rail_half_z,
            upright_half_x=self._upright_half_x,
            upright_half_y=self._upright_half_y,
            upright_local_x=self._upright_local_x,
            chamfer=self._bin_chamfer,
        )
        pallet_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._plate_half_thickness + 0.001])
        self.pallet = pallet_builder.build(name="arch_pallet")
        self.belt.register_rider(self.pallet, patch_radius=self._pallet_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        specs = [spec_by_id[name] for name in self._part_names]
        self.part = self._build_part_group(specs, "clear_arch_part", deck_z + 0.50)
        self.belt.register_rider(
            self.part,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._part_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self._part_half_z = torch.as_tensor([s.bbox[2] for s in specs], dtype=torch.float32, device=self.device)
        self._part_half_fp = torch.as_tensor(
            [max(s.bbox[0], s.bbox[1]) for s in specs], dtype=torch.float32, device=self.device
        )
        self._part_grasp_half = torch.as_tensor(
            [(s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1])) / 2.0 for s in specs],
            dtype=torch.float32,
            device=self.device,
        )

    def _build_part_group(self, specs: list[ObjectSpec], group: str, park_z: float) -> Actor:
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
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"put the {name} in the pallet bin and pull your hand clear of the arch" for name in self._part_display]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = self._part_grasp_half[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_arch_trackers(env_idx)

        pallet_pos = self._sample_pallet_position(env_idx, b)
        pallet_q = self._sample_pallet_orientation(env_idx, b)
        self.pallet.set_pose(Pose.create_from_pq(pallet_pos, pallet_q))
        pallet_vel = torch.zeros((b, 3), device=device)
        pallet_vel[:, 1] = self.belt.velocity[env_idx]
        self.pallet.set_linear_velocity(pallet_vel)
        self.pallet.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        part_q = torch.zeros((b, 4), device=device)
        part_q[:, 0] = 1.0
        self.part.set_pose(Pose.create_from_pq(tcp_pos, part_q))
        self.part.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.part.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _sample_pallet_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        base = (1.0 - self.belt.direction_sign[env_idx]) * (math.pi / 2.0)
        yaw = base
        if self.randomization.pallet_yaw:
            lo, hi = self.randomization.pallet_yaw_range
            jitter = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            yaw = yaw + jitter
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(yaw * 0.5)
        q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_pallet_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length + self._plate_back_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._plate_half_thickness + 0.001
        return pos

    def _reset_arch_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_clear_sustain_counter"):
            self._clear_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._arch_struck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._clear_sustain_counter[env_idx] = 0
        self._arch_struck[env_idx] = False

    def _hand_probe_points(self) -> list[torch.Tensor]:
        agent = cast("Panda", self.agent)
        return [agent.tcp.pose.p, agent.finger1_link.pose.p, agent.finger2_link.pose.p]

    def in_arch_corridor(self, world: torch.Tensor) -> torch.Tensor:
        local = self._in_pallet_frame(world)
        near_arch = torch.abs(local[:, 1] + self._arch_offset) <= self._slab_half_y
        over_belt = torch.abs(local[:, 0]) <= self._slab_half_x
        under_rail = local[:, 2] <= self._rail_top_local_z + self._slab_top_margin
        return near_arch & over_belt & under_rail

    def arch_span_clearance(self, world: torch.Tensor) -> torch.Tensor:
        local = self._in_pallet_frame(world)
        lateral = torch.abs(local[:, 0]) - self._slab_half_x
        vertical = local[:, 2] - (self._rail_top_local_z + self._slab_top_margin)
        return torch.maximum(lateral, vertical)

    def evaluate(self):
        agent = cast("Panda", self.agent)
        pallet_pose = self.pallet.pose
        pallet_pos = pallet_pose.p
        part_pos = self.part.pose.p
        tcp_pos = agent.tcp.pose.p
        belt_v = self.belt.velocity
        sign = self.belt.direction_sign

        part_local = self._in_pallet_frame(part_pos)
        reach = self._bin_inner_half - self._part_half_fp - self._containment_margin
        inside_xy = (torch.abs(part_local[:, 0]) <= reach) & (torch.abs(part_local[:, 1]) <= reach)
        below_rim = (part_local[:, 2] - self._part_half_z) <= (self._bin_rim_local_z - self._below_rim_depth)
        contained = inside_xy & below_rim

        finger_width = self.gripper_aperture()
        released = finger_width >= self._released_finger_width

        part_v = self.part.linear_velocity
        settled = (torch.abs(part_v[:, 1] - belt_v) < self._comove_tol) & (torch.abs(part_v[:, 0]) < self._comove_tol)

        hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for point in self._hand_probe_points():
            hit = hit | self.in_arch_corridor(point)
        self._arch_struck = self._arch_struck | hit
        no_strike = ~self._arch_struck

        arch_pos = self.arch_world_pos()
        arch_past_tcp = (arch_pos[:, 1] - tcp_pos[:, 1]) * sign
        deadline = arch_past_tcp >= self._deadline_clear_dist

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        pallet_up = torch.zeros((self.num_envs, 3), device=self.device)
        pallet_up[:, 2] = 1.0
        tilt_cos = (self._to_world(pallet_pose, pallet_up) - pallet_pos)[:, 2]
        pallet_on_belt = (
            (torch.abs(pallet_pos[:, 0] - ox) <= self._pallet_x_tol)
            & (torch.abs(self.pallet.linear_velocity[:, 1] - belt_v) <= self._pallet_comove_tol)
            & (tilt_cos >= self._pallet_tilt_cos)
            & (pallet_pos[:, 2] > deck_z)
            & (torch.abs(pallet_pos[:, 1] - oy) <= self._conveyor_half_length)
        )

        cleared = contained & released & settled & no_strike & deadline & pallet_on_belt

        self._clear_sustain_counter = self._advance_sustain("clear_sustain", self._clear_sustain_counter, cleared)
        success = self._clear_sustain_counter >= self._clear_sustain_steps

        return {
            "success": success,
            "contained": contained,
            "released": released,
            "settled": settled,
            "no_strike": no_strike,
            "deadline": deadline,
            "pallet_on_belt": pallet_on_belt,
            "clear_sustain": self._clear_sustain_counter,
            "arch_past_tcp": arch_past_tcp,
            "tcp_to_part": torch.linalg.norm(tcp_pos - part_pos, dim=-1),
            "part_to_bin_xy": torch.linalg.norm(part_local[:, :2], dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        part_pos = self.part.pose.p
        bin_pos = self.pallet.pose.p

        xy_dist = torch.linalg.norm(part_pos[:, :2] - bin_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        loaded = (info["contained"] & info["released"]).float()
        evacuated = (info["contained"] & info["no_strike"] & info["deadline"]).float()
        return carry_reward + loaded + 2.0 * evacuated

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
