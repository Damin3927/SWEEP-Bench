from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_bin
from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_kit_fill.assets import build_kit_tray
from sweep_bench.tasks.conveyor_kit_fill.config import RandomizationConfig
from sweep_bench.tasks.conveyor_kit_fill.scripted import ScriptedKitFillMixin
from sweep_bench.tasks.conveyor_kit_fill.variants import KIT_FILL_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec

_HOLDER_COLOR = (0.46, 0.42, 0.36, 1.0)


@bench_task("ConveyorKitFill-v0")
@register_env("ConveyorKitFill-v0", max_episode_steps=300)
class ConveyorKitFillEnv(ScriptedKitFillMixin, ConveyorWorkspaceEnv):
    """Put the part into the EMPTY cell of a multi-cell kit tray riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(KIT_FILL_OBJECT_IDS)

    _cell_count: int = 3
    _cell_inner_half: float = 0.035
    _wall_half_thickness: float = 0.003
    _wall_half_height: float = 0.006
    _floor_half_height: float = 0.004
    _cell_chamfer: float = 0.004
    _default_empty_cell: int = 0

    _tray_patch_radius: float = 0.035
    _part_patch_radius: float = 0.015

    _holder_inner_half: float = 0.075
    _holder_wall_thickness: float = 0.006
    _holder_wall_half_height: float = 0.004
    _finger_open_outer_half: float = 0.056
    _finger_below_tcp: float = 0.009
    _holder_offset_x: float = -0.30
    _holder_offset_y_mag: float = 0.22

    _cell_margin: float = 0.010
    _below_rim_depth: float = 0.005
    _misfeed_stack_clearance: float = 0.030
    _released_finger_width: float = 0.035
    _released_distance_threshold: float = 0.055
    _comove_tol: float = 0.06
    _tray_comove_tol: float = 0.08
    _tray_x_tol: float = 0.04
    _tray_tilt_cos: float = math.cos(math.radians(10.0))
    _drop_below_deck: float = 0.02
    _drop_x_band: float = 0.15
    _fill_sustain_steps: int = 5

    default_prompt: str = "put the part into the empty cell of the kit tray on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        halves = [max(spec_by_id[o].bbox[0], spec_by_id[o].bbox[1]) for o in self.objects]
        widest, narrowest = max(halves), min(halves)

        if widest >= self._cell_inner_half:
            raise ValueError(
                f"part half-extent {widest} m does not fit the cell's clear half-width {self._cell_inner_half} m"
            )
        if narrowest < self._cell_margin:
            raise ValueError(
                f"part half-extent {narrowest} m is under the cell centring margin {self._cell_margin} m: "
                f"a correctly seated part could sit outside the predicate's cell reach"
            )
        if 2.0 * widest >= self._released_finger_width:
            raise ValueError(
                f"part width {2.0 * widest} m is not below the released finger aperture "
                f"{self._released_finger_width} m: a gripping hand would read as open"
            )

        if not self._part_in_holder:
            return
        if self._holder_inner_half <= self._finger_open_outer_half:
            raise ValueError(
                f"feeder holder interior half-width {self._holder_inner_half} m is too small for the OPEN hand: "
                f"fingertip outer reach is {self._finger_open_outer_half} m"
            )
        shortest = min(spec_by_id[o].bbox[2] for o in self.objects)
        if shortest <= self._finger_below_tcp:
            raise ValueError(
                f"part half-height {shortest} m is not above the fingertip drop below the TCP "
                f"({self._finger_below_tcp} m): the pick would drive the fingertips into the holder floor"
            )

    @property
    def _part_in_holder(self) -> bool:
        return self.randomization.part_in_holder

    _holder_human_render_eye: tuple[float, float, float] = (0.75, 0.0, 1.25)
    _holder_human_render_target: tuple[float, float, float] = (-0.1, 0.0, 0.15)

    @property
    def _default_human_render_camera_configs(self):
        if not self._part_in_holder:
            return super()._default_human_render_camera_configs
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=self._holder_human_render_eye,
            target=self._holder_human_render_target,
        )

    @property
    def _cell_pitch(self) -> float:
        return 2.0 * (self._cell_inner_half + self._wall_half_thickness)

    @property
    def _tray_outer_half_x(self) -> float:
        return self._cell_inner_half + 2.0 * self._wall_half_thickness

    @property
    def _tray_outer_half_y(self) -> float:
        return self._cell_count * self._cell_inner_half + (self._cell_count + 1) * self._wall_half_thickness

    @property
    def _cell_floor_local_z(self) -> float:
        return self._floor_half_height

    @property
    def _cell_rim_local_z(self) -> float:
        return self._floor_half_height + 2.0 * self._wall_half_height

    @property
    def _cell_mouth_local_z(self) -> float:
        return self._cell_rim_local_z + self._cell_chamfer

    @property
    def _holder_floor_half_height(self) -> float:
        return self._conveyor_deck_height / 2.0

    @property
    def _holder_origin_above_surface(self) -> float:
        return self._holder_floor_half_height

    def _cell_local_y(self, cell: torch.Tensor) -> torch.Tensor:
        return (cell.to(torch.float32) - (self._cell_count - 1) / 2.0) * self._cell_pitch

    def _cell_local_offset(self, cell: torch.Tensor) -> torch.Tensor:
        offset = torch.zeros((cell.shape[0], 3), device=self.device)
        offset[:, 1] = self._cell_local_y(cell)
        return offset

    def _to_world(self, tray_pose: Pose, local: torch.Tensor) -> torch.Tensor:
        return (tray_pose * Pose.create_from_pq(local)).p

    def _in_tray_frame(self, world: torch.Tensor) -> torch.Tensor:
        return (self.tray.pose.inv() * Pose.create_from_pq(world)).p

    def target_cell_offset_local(self) -> torch.Tensor:
        return self._cell_local_offset(self._empty_cell)

    def target_cell_world_pos(self) -> torch.Tensor:
        return self._to_world(self.tray.pose, self.target_cell_offset_local())

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        tray_builder = build_kit_tray(
            self.scene,
            n_cells=self._cell_count,
            cell_inner_half=self._cell_inner_half,
            wall_half_thickness=self._wall_half_thickness,
            wall_half_height=self._wall_half_height,
            floor_half_height=self._floor_half_height,
            chamfer=self._cell_chamfer,
        )
        tray_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._floor_half_height + 0.001])
        self.tray = tray_builder.build(name="kit_tray")
        self.belt.register_rider(self.tray, patch_radius=self._tray_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        specs = [spec_by_id[name] for name in self._part_names]

        self.part = self._build_part_group(specs, "kit_part", deck_z + 0.50)
        self.fillers: list[Actor] = [
            self._build_part_group(specs, f"kit_filler_{j}", deck_z + 0.60 + 0.06 * j)
            for j in range(self._cell_count - 1)
        ]
        self.belt.register_rider(
            self.part,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._part_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self._part_half_z = torch.as_tensor([s.bbox[2] for s in specs], dtype=torch.float32, device=self.device)
        self._part_grasp_half = torch.as_tensor(
            [(s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1])) / 2.0 for s in specs],
            dtype=torch.float32,
            device=self.device,
        )

        if not self._part_in_holder:
            return
        holder_builder = build_bin(
            self.scene,
            inner_half_x=self._holder_inner_half,
            inner_half_y=self._holder_inner_half,
            wall_thickness=self._holder_wall_thickness,
            wall_half_height=self._holder_wall_half_height,
            floor_half_height=self._holder_floor_half_height,
            color=_HOLDER_COLOR,
        )
        holder_builder.initial_pose = sapien.Pose(
            p=[ox + self._holder_offset_x, oy, oz + self._holder_origin_above_surface]
        )
        self.holder = holder_builder.build_kinematic(name="feeder_holder")

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
        return [
            f"put the {name} into the empty cell of the kit tray on the conveyor belt" for name in self._part_display
        ]

    def _rest_aperture(self, env_idx: torch.Tensor) -> float | np.ndarray:
        if self._part_in_holder:
            return super()._rest_aperture(env_idx)
        half_w = self._part_grasp_half[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, *self.fillers]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_fill_trackers(env_idx)
        self._empty_cell[env_idx] = self._sample_empty_cell(b)

        tray_pos = self._sample_tray_position(env_idx, b)
        tray_q = self._sample_tray_orientation(b)
        tray_pose = Pose.create_from_pq(tray_pos, tray_q)
        self.tray.set_pose(tray_pose)
        tray_vel = torch.zeros((b, 3), device=device)
        tray_vel[:, 1] = self.belt.velocity[env_idx]
        self.tray.set_linear_velocity(tray_vel)
        self.tray.set_angular_velocity(torch.zeros((b, 3), device=device))

        empty = self._empty_cell[env_idx]
        part_half_z = self._part_half_z[env_idx]
        for j, filler in enumerate(self.fillers):
            cell = j + (empty <= j).long()
            self._filler_cells[env_idx, j] = cell
            local = self._cell_local_offset(cell)
            local[:, 2] = self._cell_floor_local_z + part_half_z + 0.0005
            filler.set_pose(Pose.create_from_pq(self._to_world(tray_pose, local), tray_q))
            filler.set_linear_velocity(tray_vel)
            filler.set_angular_velocity(torch.zeros((b, 3), device=device))

        part_q = torch.zeros((b, 4), device=device)
        part_q[:, 0] = 1.0
        if self._part_in_holder:
            holder_pos = self._sample_holder_position(env_idx, b)
            self.holder.set_pose(Pose.create_from_pq(holder_pos, part_q))
            part_pos = holder_pos.clone()
            part_pos[:, 2] = self.workspace_offset[2] + 2.0 * self._holder_floor_half_height + part_half_z + 0.001
        else:
            if self.scene.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()
            part_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        self.part.set_pose(Pose.create_from_pq(part_pos, part_q))
        self.part.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.part.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _sample_empty_cell(self, b: int) -> torch.Tensor:
        if self.randomization.empty_cell_index:
            picks = self._episode_rng.choice(np.arange(self._cell_count), size=b)
            return torch.as_tensor(picks, dtype=torch.long, device=self.device)
        return torch.full((b,), self._default_empty_cell, dtype=torch.long, device=self.device)

    def _sample_tray_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.tray_yaw:
            lo, hi = self.randomization.tray_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_tray_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._tray_outer_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._floor_half_height + 0.001
        return pos

    def _sample_holder_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        rcfg = self.randomization
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox + self._holder_offset_x
        pos[:, 1] = oy + self._holder_offset_y_mag * self.belt.direction_sign[env_idx]
        pos[:, 2] = oz + self._holder_origin_above_surface
        if rcfg.feeder_position:
            dx = self._episode_rng.uniform(*rcfg.feeder_offset_range, size=b).astype(np.float32)
            dy = self._episode_rng.uniform(*rcfg.feeder_offset_y_range, size=b).astype(np.float32)
            pos[:, 0] += torch.as_tensor(dx, device=device)
            pos[:, 1] += torch.as_tensor(dy, device=device)
        return pos

    def _reset_fill_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_fill_sustain_counter"):
            self._fill_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._empty_cell = torch.full(
                (self.num_envs,), self._default_empty_cell, dtype=torch.long, device=self.device
            )
            self._filler_cells = torch.zeros(
                (self.num_envs, self._cell_count - 1), dtype=torch.long, device=self.device
            )
            self._misfeed_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fill_sustain_counter[env_idx] = 0
        self._misfeed_latch[env_idx] = False

    def _inside_cell_xy(self, local: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        reach = self._cell_inner_half - self._cell_margin
        return (torch.abs(local[:, 0]) <= reach) & (torch.abs(local[:, 1] - self._cell_local_y(cell)) <= reach)

    def _seated_in_cell(self, local: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        below_rim = (local[:, 2] - self._part_half_z) <= (self._cell_rim_local_z - self._below_rim_depth)
        return self._inside_cell_xy(local, cell) & below_rim

    def evaluate(self):
        agent = cast("Panda", self.agent)
        tray_pose = self.tray.pose
        tray_pos = tray_pose.p
        part_pos = self.part.pose.p
        belt_v = self.belt.velocity

        part_local = self._in_tray_frame(part_pos)
        in_target_cell = self._seated_in_cell(part_local, self._empty_cell)

        fillers_intact = torch.ones_like(in_target_cell)
        for j, filler in enumerate(self.fillers):
            filler_local = self._in_tray_frame(filler.pose.p)
            fillers_intact &= self._seated_in_cell(filler_local, self._filler_cells[:, j])

        finger_width = self.gripper_aperture()
        hand_open = finger_width >= self._released_finger_width
        released = hand_open & (
            torch.linalg.norm(agent.tcp.pose.p - part_pos, dim=-1) > self._released_distance_threshold
        )

        part_v = self.part.linear_velocity
        comoving = (torch.abs(part_v[:, 1] - belt_v) < self._comove_tol) & (torch.abs(part_v[:, 0]) < self._comove_tol)

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        tray_up = torch.zeros((self.num_envs, 3), device=self.device)
        tray_up[:, 2] = 1.0
        tilt_cos = (self._to_world(tray_pose, tray_up) - tray_pos)[:, 2]
        tray_on_belt = (
            (torch.abs(tray_pos[:, 0] - ox) <= self._tray_x_tol)
            & (torch.abs(self.tray.linear_velocity[:, 1] - belt_v) <= self._tray_comove_tol)
            & (tilt_cos >= self._tray_tilt_cos)
            & (tray_pos[:, 2] > deck_z)
            & (torch.abs(tray_pos[:, 1] - oy) < self._conveyor_half_length + self._tray_outer_half_y)
        )

        in_filled_cell = torch.zeros_like(in_target_cell)
        for j in range(self._cell_count - 1):
            in_filled_cell |= self._inside_cell_xy(part_local, self._filler_cells[:, j])
        on_the_cell = (part_local[:, 2] - self._part_half_z) <= (
            self._cell_mouth_local_z + self._misfeed_stack_clearance
        )
        unheld = ~agent.is_grasping(self.part)
        in_filled_cell = in_filled_cell & unheld & on_the_cell
        off_the_line = (part_pos[:, 2] < deck_z - self._drop_below_deck) & (
            torch.abs(part_pos[:, 0] - ox) <= self._drop_x_band
        )
        self._misfeed_latch = self._misfeed_latch | in_filled_cell | off_the_line
        no_misfeed = ~self._misfeed_latch

        filled = in_target_cell & fillers_intact & released & comoving & tray_on_belt & no_misfeed

        self._fill_sustain_counter = self._advance_sustain("fill_sustain", self._fill_sustain_counter, filled)
        success = self._fill_sustain_counter >= self._fill_sustain_steps

        return {
            "success": success,
            "in_target_cell": in_target_cell,
            "fillers_intact": fillers_intact,
            "released": released,
            "comoving": comoving,
            "tray_on_belt": tray_on_belt,
            "no_misfeed": no_misfeed,
            "in_filled_cell": in_filled_cell,
            "off_the_line": off_the_line,
            "fill_sustain": self._fill_sustain_counter,
            "part_to_cell_xy": torch.linalg.norm(
                part_local[:, :2] - self._cell_local_offset(self._empty_cell)[:, :2], dim=-1
            ),
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        part_pos = self.part.pose.p
        cell_pos = self.target_cell_world_pos()

        xy_dist = torch.linalg.norm(part_pos[:, :2] - cell_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        seated = (info["in_target_cell"] & info["fillers_intact"] & info["no_misfeed"]).float()
        if not self._part_in_holder:
            return carry_reward + 2.0 * seated
        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(agent.tcp.pose.p - part_pos, dim=-1))
        return reach_reward + carry_reward + 2.0 * seated

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / (4.0 if self._part_in_holder else 3.0)
