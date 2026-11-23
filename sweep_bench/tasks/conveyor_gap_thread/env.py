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
from sweep_bench.tasks.conveyor_gap_thread.assets import build_divider, build_grooved_crate
from sweep_bench.tasks.conveyor_gap_thread.config import RandomizationConfig
from sweep_bench.tasks.conveyor_gap_thread.scripted import ScriptedGapThreadMixin
from sweep_bench.tasks.conveyor_gap_thread.variants import GAP_THREAD_OBJECT_IDS, GAP_THREAD_RIDER_SPECS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec


@bench_task("ConveyorGapThread-v0")
@register_env("ConveyorGapThread-v0", max_episode_steps=300)
class ConveyorGapThreadEnv(ScriptedGapThreadMixin, ConveyorWorkspaceEnv):
    """Lower the held rider into the crate's slot without disturbing the dividers."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(GAP_THREAD_OBJECT_IDS, extra=GAP_THREAD_RIDER_SPECS)

    _base_half_x: float = 0.060
    _base_half_y: float = 0.065
    _base_half_z: float = 0.003
    _groove_depth: float = 0.016
    _groove_clearance: float = 0.0003
    _wall_half_thickness: float = 0.005
    _wall_half_height: float = 0.012

    _divider_half_thickness: float = 0.004
    _divider_half_length: float = 0.050
    _divider_half_height: float = 0.025
    _divider_density: float = 1000.0

    _default_slot_width: float = 0.085

    _crate_patch_radius: float = 0.045
    _rider_patch_radius: float = 0.015

    _slot_margin: float = 0.002
    _seated_z_tol: float = 0.006
    _released_finger_width: float = 0.035
    _released_distance_threshold: float = 0.07
    _comove_tol: float = 0.06
    _divider_drift_tol: float = 0.004
    _divider_tilt_deg: float = 4.0
    _crate_z_tol: float = 0.006
    _crate_x_tol: float = 0.04
    _thread_sustain_steps: int = 8

    default_prompt: str = "lower the object into the slot of the crate without touching the dividers"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        slot_width: float | None = None,
        **kwargs,
    ) -> None:
        self._slot_width: float = self._default_slot_width if slot_width is None else float(slot_width)
        super().__init__(*args, **kwargs)
        self._assert_slot_hosts_catalog()

    def _assert_slot_hosts_catalog(self) -> None:
        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        used = [spec_by_id[name] for name in self.objects]
        max_half_fp = max(max(spec.bbox[0], spec.bbox[1]) for spec in used)
        if max_half_fp + self._slot_margin >= self._slot_half:
            raise ValueError(
                f"widest adopted rider half-footprint {max_half_fp:.4f} m leaves no room in a "
                f"{2 * self._slot_half:.3f} m slot at margin {self._slot_margin}: success clause (a) would be "
                f"unreachable for it."
            )

        min_half_z = min(spec.bbox[2] for spec in used)
        thread_depth = min_half_z - self._finger_below_tcp - self._finger_divider_clearance
        if thread_depth < self._min_thread_depth:
            raise ValueError(
                f"shortest adopted rider half-height {min_half_z:.4f} m can only be threaded "
                f"{thread_depth:.4f} m past the divider tops with the jaws clear of them, under the "
                f"{self._min_thread_depth} m this family's corridor needs: pick a taller rider (see "
                f"`variants.py` for why the vetted catalog has none)."
            )

        rcfg = self.randomization
        yaw = max(abs(v) for v in rcfg.crate_yaw_range) if getattr(rcfg, "crate_yaw", False) else 0.0
        jaw_reach = self._finger_half_across * math.cos(yaw) + self._finger_open_reach * math.sin(yaw)
        if jaw_reach >= self._slot_half:
            raise ValueError(
                f"open jaw reaches {jaw_reach:.4f} m across the slot axis at {math.degrees(yaw):.1f} deg of crate "
                f"yaw, past the {self._slot_half:.4f} m slot half-width: the release would drive a finger pad "
                f"into a divider."
            )

    _finger_half_across: float = 0.011
    _finger_open_reach: float = 0.066
    _finger_below_tcp: float = 0.0095
    _finger_divider_clearance: float = 0.002
    _min_thread_depth: float = 0.015
    _min_release_floor_gap: float = 0.004

    @property
    def _slot_half(self) -> float:
        return self._slot_width / 2.0

    @property
    def _groove_center_y(self) -> float:
        return self._slot_half + self._divider_half_thickness

    @property
    def _groove_half_width(self) -> float:
        return self._divider_half_thickness + self._groove_clearance

    @property
    def _groove_floor_local_z(self) -> float:
        return self._base_half_z

    @property
    def _crate_floor_local_z(self) -> float:
        return self._base_half_z + self._groove_depth

    @property
    def _release_floor_gap(self) -> torch.Tensor:
        exposed = self._divider_top_local_z - self._crate_floor_local_z
        gap = exposed - self._rider_half_z + self._finger_below_tcp + self._finger_divider_clearance
        return gap.clamp(min=self._min_release_floor_gap)

    @property
    def _divider_top_local_z(self) -> float:
        return self._groove_floor_local_z + 2.0 * self._divider_half_height

    @property
    def _divider_home_local(self) -> torch.Tensor:
        gy, tz = self._groove_center_y, self._divider_top_local_z
        return torch.as_tensor([[0.0, gy, tz], [0.0, -gy, tz]], dtype=torch.float32, device=self.device)

    def _in_crate_frame(self, world: torch.Tensor) -> torch.Tensor:
        return (self.crate.pose.inv() * Pose.create_from_pq(world)).p

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        crate_builder = build_grooved_crate(
            self.scene,
            base_half_x=self._base_half_x,
            base_half_y=self._base_half_y,
            base_half_z=self._base_half_z,
            pad_half_z=self._groove_depth / 2.0,
            groove_center_y=self._groove_center_y,
            groove_half_width=self._groove_half_width,
            wall_half_thickness=self._wall_half_thickness,
            wall_half_height=self._wall_half_height,
        )
        crate_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._base_half_z + 0.001])
        self.crate = crate_builder.build(name="gap_thread_crate")
        self.belt.register_rider(self.crate, patch_radius=self._crate_patch_radius)

        self.dividers: list[Actor] = []
        for i, side in enumerate((1.0, -1.0)):
            builder = build_divider(
                self.scene,
                half_length=self._divider_half_length,
                half_thickness=self._divider_half_thickness,
                half_height=self._divider_half_height,
                density=self._divider_density,
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy + side * self._groove_center_y, deck_z + 0.30 + 0.06 * i])
            self.dividers.append(builder.build(name=f"gap_thread_divider_{i}"))

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._rider_names: list[str] = [str(name) for name in names]
        self._rider_display: list[str] = [display_name(spec_by_id[name]) for name in self._rider_names]
        specs: list[ObjectSpec] = [spec_by_id[name] for name in self._rider_names]
        self.rider = self._build_rider_group(specs, deck_z + 0.50)
        self.belt.register_rider(
            self.rider,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._rider_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self._rider_half_z = torch.as_tensor([s.bbox[2] for s in specs], dtype=torch.float32, device=self.device)
        self._rider_half_fp = torch.as_tensor(
            [max(s.bbox[0], s.bbox[1]) for s in specs], dtype=torch.float32, device=self.device
        )
        self._rider_grasp_half = torch.as_tensor(
            [(s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1])) / 2.0 for s in specs],
            dtype=torch.float32,
            device=self.device,
        )

    def _build_rider_group(self, specs: list[ObjectSpec], park_z: float) -> Actor:
        ox, oy, _oz = self.workspace_offset
        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, park_z])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"gap_thread_rider-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        merged = Actor.merge(objs, name="gap_thread_rider")
        self.add_to_state_dict_registry(merged)
        return merged

    @property
    def episode_objects(self) -> list[str]:
        return list(self._rider_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"lower the {name} into the slot of the crate without touching the dividers" for name in self._rider_display
        ]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = self._rider_grasp_half[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, *self.dividers]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        self._reset_thread_trackers(env_idx)

        crate_pos = self._sample_crate_position(env_idx, b)
        crate_q = self._sample_crate_orientation(b)
        crate_pose = Pose.create_from_pq(crate_pos, crate_q)
        self.crate.set_pose(crate_pose)
        crate_vel = torch.zeros((b, 3), device=device)
        crate_vel[:, 1] = self.belt.velocity[env_idx]
        self.crate.set_linear_velocity(crate_vel)
        self.crate.set_angular_velocity(torch.zeros((b, 3), device=device))
        self._crate_rest_z[env_idx] = crate_pos[:, 2]

        for i, divider in enumerate(self.dividers):
            local = torch.zeros((b, 3), device=device)
            local[:, 1] = (1.0 if i == 0 else -1.0) * self._groove_center_y
            local[:, 2] = self._groove_floor_local_z + self._divider_half_height
            divider.set_pose(Pose.create_from_pq((crate_pose * Pose.create_from_pq(local)).p, crate_q))
            divider.set_linear_velocity(crate_vel)
            divider.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        rider_q = torch.zeros((b, 4), device=device)
        rider_q[:, 0] = 1.0
        self.rider.set_pose(Pose.create_from_pq(tcp_pos, rider_q))
        self.rider.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.rider.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _sample_crate_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.crate_yaw:
            lo, hi = self.randomization.crate_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_crate_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._base_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._base_half_z + 0.001
        return pos

    def _reset_thread_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_thread_sustain"):
            self._thread_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._dividers_disturbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._crate_rest_z = torch.zeros(self.num_envs, device=self.device)
        self._thread_sustain[env_idx] = 0
        self._dividers_disturbed[env_idx] = False

    def _divider_disturbance(self) -> tuple[torch.Tensor, torch.Tensor]:
        crate_pose = self.crate.pose
        crate_inv = crate_pose.inv()
        crate_up = crate_pose.to_transformation_matrix()[:, :3, 2]
        home = self._divider_home_local

        top_offset = torch.zeros((self.num_envs, 3), device=self.device)
        top_offset[:, 2] = self._divider_half_height

        drift = torch.zeros(self.num_envs, device=self.device)
        tilt_cos = torch.ones(self.num_envs, device=self.device)
        for i, divider in enumerate(self.dividers):
            pose = divider.pose
            top_world = (pose * Pose.create_from_pq(top_offset)).p
            top_local = (crate_inv * Pose.create_from_pq(top_world)).p
            drift = torch.maximum(drift, torch.linalg.norm(top_local - home[i], dim=-1))
            up = pose.to_transformation_matrix()[:, :3, 2]
            tilt_cos = torch.minimum(tilt_cos, (up * crate_up).sum(dim=-1))
        return drift, tilt_cos

    def evaluate(self):
        agent = cast("Panda", self.agent)
        ox, _oy, _oz = self.workspace_offset
        crate_pos = self.crate.pose.p
        rider_pos = self.rider.pose.p

        local = self._in_crate_frame(rider_pos)
        in_slot = torch.abs(local[:, 1]) < (self._slot_half - self._rider_half_fp - self._slot_margin)
        on_floor = torch.abs(local[:, 0]) < (self._base_half_x - self._rider_half_fp - self._slot_margin)
        rider_bottom = local[:, 2] - self._rider_half_z
        bottomed = torch.abs(rider_bottom - self._crate_floor_local_z) < self._seated_z_tol
        seated = in_slot & on_floor & bottomed

        finger_width = self.gripper_aperture()
        released = (finger_width >= self._released_finger_width) & (
            torch.linalg.norm(agent.tcp.pose.p - rider_pos, dim=-1) > self._released_distance_threshold
        )

        comoving = torch.linalg.norm(self.rider.linear_velocity - self.crate.linear_velocity, dim=-1) < self._comove_tol

        drift, tilt_cos = self._divider_disturbance()
        disturbed_now = (drift > self._divider_drift_tol) | (tilt_cos < math.cos(math.radians(self._divider_tilt_deg)))
        self._dividers_disturbed = self._dividers_disturbed | disturbed_now
        clean = ~self._dividers_disturbed

        crate_on_belt = (torch.abs(crate_pos[:, 2] - self._crate_rest_z) <= self._crate_z_tol) & (
            torch.abs(crate_pos[:, 0] - ox) <= self._crate_x_tol
        )

        threaded = seated & released & comoving & clean & crate_on_belt

        self._thread_sustain = self._advance_sustain("thread_sustain", self._thread_sustain, threaded)
        success = self._thread_sustain >= self._thread_sustain_steps

        return {
            "success": success,
            "seated": seated,
            "in_slot": in_slot,
            "on_floor": on_floor,
            "bottomed": bottomed,
            "released": released,
            "comoving": comoving,
            "clean": clean,
            "crate_on_belt": crate_on_belt,
            "divider_drift": drift,
            "divider_tilt_cos": tilt_cos,
            "thread_sustain": self._thread_sustain,
            "rider_slot_offset": torch.abs(local[:, 1]),
        }

    def compute_dense_reward(self, obs, action, info):
        rider_pos = self.rider.pose.p
        crate_pos = self.crate.pose.p

        xy_dist = torch.linalg.norm(rider_pos[:, :2] - crate_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)
        return carry_reward + 2.0 * info["seated"].float() - 2.0 * (~info["clean"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
