from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_tare_load.assets import (
    build_scale_pan,
    build_scale_well,
    build_tare_plunger,
)
from sweep_bench.tasks.conveyor_tare_load.config import RandomizationConfig
from sweep_bench.tasks.conveyor_tare_load.scripted import ScriptedTareLoadMixin
from sweep_bench.tasks.conveyor_tare_load.variants import TARE_LOAD_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec

_G = 9.81

_SPRING_DAMPING_RATIO = 1.0

_FIT_EPS = 1e-6


@bench_task("ConveyorTareLoad-v0")
@register_env("ConveyorTareLoad-v0", max_episode_steps=300)
class ConveyorTareLoadEnv(ScriptedTareLoadMixin, ConveyorWorkspaceEnv):
    """Tare the bench scale FIRST, then load the belt's part onto its pan."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(TARE_LOAD_OBJECT_IDS)

    _scale_offset_x: float = -0.28
    _scale_offset_y_mag: float = 0.265
    _station_offset_y: float = 0.085
    _scale_fast_band_speed: float = 0.25

    _pan_well_half_width: float = 0.045
    _pan_bore_half_width: float = 0.0386
    _pan_well_wall_height: float = 0.039

    _well_floor_top_above_surface: float = 0.15

    @property
    def _pan_well_floor_half_thickness(self) -> float:
        return 0.5 * self._well_floor_top_above_surface

    _pan_half_width: float = 0.038
    _pan_half_thickness: float = 0.022
    _pan_rim_half_thickness: float = 0.0015
    _pan_rim_height: float = 0.004
    _pan_rest_gap: float = 0.015

    _plunger_well_half_width: float = 0.033
    _plunger_bore_half_width: float = 0.021
    _plunger_well_wall_height: float = 0.026

    @property
    def _plunger_well_floor_half_thickness(self) -> float:
        return self._pan_well_floor_half_thickness

    _plunger_half_width: float = 0.019
    _plunger_half_height: float = 0.022
    _plunger_rest_gap: float = 0.008

    _pan_deflection_nominal: float = 0.010
    _pan_deflection_stiff: float = 0.008
    _pan_force_max: float = 20.0
    _plunger_spring_k: float = 150.0
    _plunger_spring_damping: float = 4.0
    _plunger_force_max: float = 10.0

    _part_patch_radius: float = 0.02

    _part_landing_tolerance: float = 0.010
    _part_min_half_height: float = 0.010
    _part_required_tag: str = "box"

    _tare_press_depth: float = 0.005
    _tare_sustain_steps: int = 3
    _tare_pan_empty_depth: float = 0.002
    _order_fail_depth: float = 0.004
    _load_success_depth: float = 0.006
    _resting_z_tol: float = 0.004
    _pan_footprint_inset: float = 0.008
    _settled_rel_speed_threshold: float = 0.05
    _hands_off_finger_margin: float = 0.01
    _hands_off_distance: float = 0.10
    _pan_bore_slack: float = 0.004
    _pan_seat_tol: float = 0.002
    _load_sustain_steps: int = 8

    default_prompt: str = "press the tare button, then put the object from the belt on the scale pan"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    _human_render_eye: tuple[float, float, float] = (0.75, 0.0, 1.25)
    _human_render_target: tuple[float, float, float] = (-0.1, 0.0, 0.15)

    @property
    def _default_human_render_camera_configs(self):
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=self._human_render_eye,
            target=self._human_render_target,
        )

    @property
    def _pan_well_origin_above_surface(self) -> float:
        return 2.0 * self._pan_well_floor_half_thickness

    @property
    def _plunger_well_origin_above_surface(self) -> float:
        return 2.0 * self._plunger_well_floor_half_thickness

    @property
    def _pan_anchor_local_z(self) -> float:
        return self._pan_rest_gap + self._pan_half_thickness

    @property
    def _pan_top_above_well(self) -> float:
        return self._pan_anchor_local_z + self._pan_half_thickness

    @property
    def _pan_rim_top_above_well(self) -> float:
        return self._pan_top_above_well + self._pan_rim_height

    @property
    def _pan_interior_half(self) -> float:
        return self._pan_half_width - 2.0 * self._pan_rim_half_thickness

    @property
    def _plunger_anchor_local_z(self) -> float:
        return self._plunger_rest_gap + self._plunger_half_height

    @property
    def _plunger_face_above_well(self) -> float:
        return self._plunger_anchor_local_z + self._plunger_half_height

    @property
    def _plunger_face_above_wall_top(self) -> float:
        return self._plunger_face_above_well - self._plunger_well_wall_height

    def _check_plunger_geometry(self) -> None:
        needed = self._plunger_rest_gap + self._finger_drop_below_tcp
        if self._plunger_face_above_wall_top < needed - _FIT_EPS:
            raise ValueError(
                f"tare plunger stands only {self._plunger_face_above_wall_top} m proud of its well wall; a part "
                f"pressing it to the {self._plunger_rest_gap} m floor stop needs {needed} m of clearance so the "
                f"fingertips (~{self._finger_drop_below_tcp} m below the TCP) stay above the wall top"
            )

    _finger_drop_below_tcp: float = 0.012

    def _check_part_fits(self, spec: ObjectSpec) -> None:
        if self._part_required_tag not in spec.tags:
            raise ValueError(
                f"part {spec.id!r} is not tagged {self._part_required_tag!r}; the pan has no restoring torque, so "
                f"only flat-faced parts have been screened to load it without pinning it in the bore (see "
                f"variants.py)"
            )
        footprint = max(spec.bbox[0], spec.bbox[1])
        room = self._pan_interior_half - self._part_landing_tolerance
        if footprint > room + _FIT_EPS:
            raise ValueError(
                f"part {spec.id!r} has horizontal half-extent {footprint} m; the pan leaves only {room} m inside "
                f"its rim once the landing tolerance is reserved"
            )
        if spec.bbox[2] < self._part_min_half_height - _FIT_EPS:
            raise ValueError(
                f"part {spec.id!r} is only {spec.bbox[2]} m half-tall; below {self._part_min_half_height} m the "
                f"opening fingers rake the pan rim at the release height"
            )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        self._check_plunger_geometry()

        pan_well_builder = build_scale_well(
            self.scene,
            bore_half_width=self._pan_bore_half_width,
            outer_half_width=self._pan_well_half_width,
            wall_height=self._pan_well_wall_height,
            floor_half_thickness=self._pan_well_floor_half_thickness,
        )
        pan_well_builder.initial_pose = sapien.Pose(
            p=[ox + self._scale_offset_x, oy, oz + self._pan_well_origin_above_surface]
        )
        self.pan_well = pan_well_builder.build_kinematic(name="scale_pan_well")

        plunger_well_builder = build_scale_well(
            self.scene,
            bore_half_width=self._plunger_bore_half_width,
            outer_half_width=self._plunger_well_half_width,
            wall_height=self._plunger_well_wall_height,
            floor_half_thickness=self._plunger_well_floor_half_thickness,
        )
        plunger_well_builder.initial_pose = sapien.Pose(
            p=[ox + self._scale_offset_x, oy, oz + self._plunger_well_origin_above_surface]
        )
        self.plunger_well = plunger_well_builder.build_kinematic(name="tare_plunger_well")

        pan_builder = build_scale_pan(
            self.scene,
            plate_half_width=self._pan_half_width,
            plate_half_thickness=self._pan_half_thickness,
            rim_half_thickness=self._pan_rim_half_thickness,
            rim_height=self._pan_rim_height,
        )
        pan_builder.initial_pose = sapien.Pose(
            p=[ox + self._scale_offset_x, oy, oz + self._pan_well_origin_above_surface + self._pan_anchor_local_z]
        )
        self.pan = pan_builder.build(name="scale_pan")

        plunger_builder = build_tare_plunger(
            self.scene,
            half_width=self._plunger_half_width,
            half_height=self._plunger_half_height,
        )
        plunger_builder.initial_pose = sapien.Pose(
            p=[
                ox + self._scale_offset_x,
                oy,
                oz + self._plunger_well_origin_above_surface + self._plunger_anchor_local_z,
            ]
        )
        self.plunger = plunger_builder.build(name="tare_plunger")

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        specs = [spec_by_id[name] for name in self._part_names]
        for spec in specs:
            self._check_part_fits(spec)

        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"part-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        self.part = Actor.merge(objs, name="scale_part")
        self.add_to_state_dict_registry(self.part)
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
        self._part_grasp_width = torch.as_tensor(
            [s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1]) for s in specs],
            dtype=torch.float32,
            device=self.device,
        )
        self._part_mass = torch.as_tensor([s.mass for s in specs], dtype=torch.float32, device=self.device)

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"press the tare button, then put the {name} on the scale pan" for name in self._part_display]

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.pan_well, self.plunger_well]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        oz = self.workspace_offset[2]

        plunger_xy, pan_xy = self._sample_station_xy(env_idx, b)
        upright = torch.zeros((b, 4), device=device)
        upright[:, 0] = 1.0

        pan_well_pos = torch.zeros((b, 3), device=device)
        pan_well_pos[:, :2] = pan_xy
        pan_well_pos[:, 2] = oz + self._pan_well_origin_above_surface
        self.pan_well.set_pose(Pose.create_from_pq(pan_well_pos, upright))

        plunger_well_pos = torch.zeros((b, 3), device=device)
        plunger_well_pos[:, :2] = plunger_xy
        plunger_well_pos[:, 2] = oz + self._plunger_well_origin_above_surface
        self.plunger_well.set_pose(Pose.create_from_pq(plunger_well_pos, upright))

        pan_pos = pan_well_pos.clone()
        pan_pos[:, 2] = pan_well_pos[:, 2] + self._pan_anchor_local_z
        self.pan.set_pose(Pose.create_from_pq(pan_pos, upright))
        self.pan.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.pan.set_angular_velocity(torch.zeros((b, 3), device=device))

        plunger_pos = plunger_well_pos.clone()
        plunger_pos[:, 2] = plunger_well_pos[:, 2] + self._plunger_anchor_local_z
        self.plunger.set_pose(Pose.create_from_pq(plunger_pos, upright))
        self.plunger.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.plunger.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_pan_spring(env_idx, b)

        part_pos = self._sample_part_position(env_idx, b)
        self.part.set_pose(Pose.create_from_pq(part_pos, upright))
        part_vel = torch.zeros((b, 3), device=device)
        part_vel[:, 1] = self.belt.velocity[env_idx]
        self.part.set_linear_velocity(part_vel)
        self.part.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_tare_trackers(env_idx)
        self._springs_ready = True

    def _sample_part_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._part_half_fp[env_idx] - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._part_half_z[env_idx] + 0.001
        return pos

    def _sample_scale_center(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        center = torch.zeros((b, 2), device=device)
        center[:, 0] = ox + self._scale_offset_x
        if self.randomization.scale_side:
            side = self._episode_rng.choice([-1.0, 1.0], size=b).astype(np.float32)
            center[:, 1] = oy + self._scale_offset_y_mag * torch.as_tensor(side, device=device)
        else:
            speed = self.belt.velocity[env_idx].abs()
            fast_band = (speed >= self._scale_fast_band_speed) | (speed == 0.0)
            grasp_rel = torch.where(fast_band, 1.0, -1.0)
            center[:, 1] = oy + self._scale_offset_y_mag * grasp_rel * self.belt.direction_sign[env_idx]
        return center

    def _sample_station_xy(self, env_idx: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        oy = self.workspace_offset[1]
        device = self.device
        center = self._sample_scale_center(env_idx, b)
        swap = torch.ones(b, device=device)
        if self.randomization.plunger_side:
            swap = torch.as_tensor(
                self._episode_rng.choice([-1.0, 1.0], size=b).astype(np.float32),
                device=device,
            )
        outward = torch.sign(center[:, 1] - oy)
        outward = torch.where(outward == 0.0, torch.ones_like(outward), outward)
        step = outward * self._station_offset_y * swap
        plunger_xy = center.clone()
        plunger_xy[:, 1] = center[:, 1] + step
        pan_xy = center.clone()
        pan_xy[:, 1] = center[:, 1] - step
        return plunger_xy, pan_xy

    def _pan_spring_stiffness(self, part_mass: torch.Tensor, deflection: torch.Tensor) -> torch.Tensor:
        return part_mass * _G / deflection

    def _reset_pan_spring(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        if not hasattr(self, "_pan_spring_k"):
            zeros = torch.zeros(self.num_envs, device=device)
            self._pan_spring_k = zeros.clone()
            self._pan_spring_preload = zeros.clone()
            self._pan_spring_damping = zeros.clone()

        deflection = torch.full((b,), self._pan_deflection_nominal, device=device)
        if self.randomization.spring_bin:
            stiff = self._episode_rng.uniform(0.0, 1.0, size=b) < 0.5
            deflection = torch.where(
                torch.as_tensor(stiff, device=device),
                torch.full((b,), self._pan_deflection_stiff, device=device),
                deflection,
            )

        part_mass = self._part_mass[env_idx]
        pan_mass = self.pan.mass.to(device)[env_idx]
        k = self._pan_spring_stiffness(part_mass, deflection)
        self._pan_spring_k[env_idx] = k
        self._pan_spring_preload[env_idx] = pan_mass * _G
        self._pan_spring_damping[env_idx] = 2.0 * _SPRING_DAMPING_RATIO * torch.sqrt(k * (pan_mass + part_mass))

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_scale_spring_forces()

    def _apply_scale_spring_forces(self) -> None:
        if not getattr(self, "_springs_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()

        pan_fz = (
            self._pan_spring_k * self._pan_depression()
            + self._pan_spring_preload
            - self._pan_spring_damping * self.pan.linear_velocity[:, 2]
        ).clamp(0.0, self._pan_force_max)
        plunger_fz = (
            self._plunger_spring_k * self._plunger_press_depth()
            - self._plunger_spring_damping * self.plunger.linear_velocity[:, 2]
        ).clamp(0.0, self._plunger_force_max)

        pan_force = torch.zeros((self.num_envs, 3), device=device)
        pan_force[:, 2] = pan_fz
        plunger_force = torch.zeros((self.num_envs, 3), device=device)
        plunger_force[:, 2] = plunger_fz
        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            force_buf[self.pan._body_data_index, :3] = pan_force
            force_buf[self.plunger._body_data_index, :3] = plunger_force
            self.scene.px.gpu_apply_rigid_dynamic_force()
        else:
            zero_torque = np.zeros(3, dtype=np.float32)
            self.pan._bodies[0].add_force_torque(pan_force.squeeze(0).cpu().numpy(), zero_torque)
            self.plunger._bodies[0].add_force_torque(plunger_force.squeeze(0).cpu().numpy(), zero_torque)

    def _pan_depression(self) -> torch.Tensor:
        return self._pan_anchor_local_z - (self.pan.pose.p[:, 2] - self.pan_well.pose.p[:, 2])

    def _plunger_press_depth(self) -> torch.Tensor:
        return self._plunger_anchor_local_z - (self.plunger.pose.p[:, 2] - self.plunger_well.pose.p[:, 2])

    def _reset_tare_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_tare_window"):
            self._tare_window = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._load_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._tare_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._order_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tare_window[env_idx] = 0
        self._load_sustain[env_idx] = 0
        self._tare_latched[env_idx] = False
        self._order_failed[env_idx] = False

    def _pan_seated_terms(self, pan_depression: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local = (self.pan_well.pose.inv() * self.pan.pose).p
        in_bore = (torch.abs(local[:, 0]) <= self._pan_bore_slack) & (torch.abs(local[:, 1]) <= self._pan_bore_slack)
        within_travel = pan_depression <= self._pan_rest_gap + self._pan_seat_tol
        return in_bore, within_travel

    def evaluate(self):
        agent = cast("Panda", self.agent)
        pan_pose = self.pan.pose
        pan_pos = pan_pose.p
        part_pos = self.part.pose.p
        tcp_pos = agent.tcp.pose.p

        pan_depression = self._pan_depression()
        press_depth = self._plunger_press_depth()

        plunger_pressed = press_depth >= self._tare_press_depth
        pan_empty = pan_depression < self._tare_pan_empty_depth
        tare_holding = plunger_pressed & pan_empty
        self._tare_window = self._advance_sustain("tare_window", self._tare_window, tare_holding)
        tare_now = self._tare_window >= self._tare_sustain_steps

        self._order_failed = self._order_failed | (~self._tare_latched & (pan_depression > self._order_fail_depth))
        self._tare_latched = self._tare_latched | tare_now

        weighed = pan_depression >= self._load_success_depth

        local = (pan_pose.inv() * Pose.create_from_pq(part_pos)).p
        bottom_gap = torch.abs((local[:, 2] - self._part_half_z) - self._pan_half_thickness)
        reach = self._pan_half_width - self._pan_footprint_inset
        resting = (
            (bottom_gap <= self._resting_z_tol) & (torch.abs(local[:, 0]) <= reach) & (torch.abs(local[:, 1]) <= reach)
        )

        rel_speed = torch.linalg.norm(self.part.linear_velocity - self.pan.linear_velocity, dim=-1)
        settled = rel_speed < self._settled_rel_speed_threshold

        finger_width = self.gripper_aperture()
        hands_off = (
            (finger_width > self._part_grasp_width + self._hands_off_finger_margin)
            & (torch.linalg.norm(tcp_pos - part_pos, dim=-1) > self._hands_off_distance)
            & (torch.linalg.norm(tcp_pos - pan_pos, dim=-1) > self._hands_off_distance)
        )

        in_bore, within_travel = self._pan_seated_terms(pan_depression)
        pan_seated = in_bore & within_travel

        loading = weighed & resting & settled & hands_off & pan_seated & self._tare_latched
        self._load_sustain = self._advance_sustain("load_sustain", self._load_sustain, loading)
        success = (self._load_sustain >= self._load_sustain_steps) & ~self._order_failed

        return {
            "success": success,
            "tare_latched": self._tare_latched,
            "order_failed": self._order_failed,
            "plunger_pressed": plunger_pressed,
            "pan_empty": pan_empty,
            "tare_window": self._tare_window,
            "weighed": weighed,
            "resting": resting,
            "settled": settled,
            "hands_off": hands_off,
            "pan_in_bore": in_bore,
            "pan_within_travel": within_travel,
            "pan_depression": pan_depression,
            "press_depth": press_depth,
            "load_sustain": self._load_sustain,
            "part_to_pan_xy": torch.linalg.norm(local[:, :2], dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p
        pan_pos = self.pan.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - part_pos, dim=-1))
        tare_reward = torch.where(
            info["tare_latched"],
            torch.ones_like(info["press_depth"]),
            (info["press_depth"] / self._tare_press_depth).clamp(0.0, 1.0),
        )
        carry_reward = (1 - torch.tanh(8 * torch.linalg.norm(part_pos[:, :2] - pan_pos[:, :2], dim=-1))) * info[
            "tare_latched"
        ].float()
        weigh_reward = (info["pan_depression"] / self._load_success_depth).clamp(0.0, 1.0)

        return reach_reward + tare_reward + carry_reward + 2.0 * weigh_reward * info["resting"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
