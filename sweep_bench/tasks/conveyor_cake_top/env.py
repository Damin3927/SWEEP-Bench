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
from sweep_bench.tasks.conveyor_cake_top.assets import build_cake_blank, crown_disc_geometry
from sweep_bench.tasks.conveyor_cake_top.config import RandomizationConfig
from sweep_bench.tasks.conveyor_cake_top.scripted import ScriptedCakeTopMixin
from sweep_bench.tasks.conveyor_cake_top.variants import CAKE_TOP_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorCakeTop-v0")
@register_env("ConveyorCakeTop-v0", max_episode_steps=300)
class ConveyorCakeTopEnv(ScriptedCakeTopMixin, ConveyorWorkspaceEnv):
    """Set the held topping inside the printed zone on a moving cake blank."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(CAKE_TOP_OBJECT_IDS)

    _cake_radius: float = 0.065
    _cake_body_half_z: float = 0.0135
    _crown_rise: float = 0.003
    _crown_discs: int = 3
    _crown_plateau_radius: float = 0.050

    _zone_radius: float = 0.030
    _zone_offset_radius: float = 0.032
    _zone_offset_angle: float = 0.0

    _cake_patch_radius: float = 0.045
    _topping_patch_radius: float = 0.015

    _rest_band: float = 0.004
    _released_finger_width: float = 0.035
    _released_distance_threshold: float = 0.06
    _comove_tol_y: float = 0.06
    _comove_tol_x: float = 0.05
    _pair_comove_tol: float = 0.05
    _cake_x_tol: float = 0.05
    _cake_comove_tol: float = 0.08
    _cake_tilt_cos: float = math.cos(math.radians(10.0))
    _cake_y_limit: float = 0.85
    _top_sustain_steps: int = 10

    default_prompt: str = "put the topping inside the printed target zone on the cake on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        objects: tuple[str, ...] | None = None,
        **kwargs,
    ) -> None:
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        if reconfiguration_freq is None:
            load_time_axes = (
                (objects is not None and len(objects) > 1) or randomization.zone_offset or randomization.zone_radius
            )
            reconfiguration_freq = 1 if (num_envs == 1 and load_time_axes) else 0
        super().__init__(
            *args,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            randomization=randomization,
            objects=objects,
            **kwargs,
        )

    @property
    def _cake_height(self) -> float:
        return 2.0 * self._cake_body_half_z + self._crown_rise

    @property
    def _crown_top_local_z(self) -> float:
        return self._cake_body_half_z + self._crown_rise

    @property
    def _crown_discs_geometry(self) -> tuple[tuple[float, float], ...]:
        return crown_disc_geometry(
            radius=self._cake_radius,
            body_half_z=self._cake_body_half_z,
            crown_rise=self._crown_rise,
            crown_discs=self._crown_discs,
            plateau_radius=self._crown_plateau_radius,
        )

    def _crown_surface_local_z(self, footprint_inner_radius: torch.Tensor) -> torch.Tensor:
        surface = torch.full_like(footprint_inner_radius, self._cake_body_half_z)
        for disc_radius, disc_top in self._crown_discs_geometry:
            surface = torch.where(footprint_inner_radius <= disc_radius, torch.full_like(surface, disc_top), surface)
        return surface

    def _in_cake_frame(self, world: torch.Tensor) -> torch.Tensor:
        return (self.cake.pose.inv() * Pose.create_from_pq(world)).p

    def zone_offset_local(self) -> torch.Tensor:
        return self._zone_offset

    def zone_world_pos(self) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, :2] = self.zone_offset_local()
        local[:, 2] = self._crown_surface_local_z(torch.linalg.norm(local[:, :2], dim=-1))
        return (self.cake.pose * Pose.create_from_pq(local)).p

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        b = self.num_envs

        rcfg = self.randomization
        rng = self._batched_episode_rng
        if rcfg.zone_offset:
            lo, hi = rcfg.zone_offset_range
            radii = np.asarray(rng.uniform(lo, hi), dtype=np.float32)
            angles = np.asarray(rng.uniform(-math.pi, math.pi), dtype=np.float32)
        else:
            radii = np.full(b, self._zone_offset_radius, dtype=np.float32)
            angles = np.full(b, self._zone_offset_angle, dtype=np.float32)
        if rcfg.zone_radius:
            lo, hi = rcfg.zone_radius_range
            zone_radii = np.asarray(rng.uniform(lo, hi), dtype=np.float32)
        else:
            zone_radii = np.full(b, self._zone_radius, dtype=np.float32)
        offsets = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)
        self._zone_offset = torch.as_tensor(offsets, dtype=torch.float32, device=device)
        self._zone_radius_t = torch.as_tensor(zone_radii, dtype=torch.float32, device=device)

        cakes: list[Actor] = []
        for i in range(b):
            builder = build_cake_blank(
                self.scene,
                radius=self._cake_radius,
                body_half_z=self._cake_body_half_z,
                crown_rise=self._crown_rise,
                crown_discs=self._crown_discs,
                plateau_radius=self._crown_plateau_radius,
                zone_offset=(float(offsets[i, 0]), float(offsets[i, 1])),
                zone_radius=float(zone_radii[i]),
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._cake_body_half_z + 0.001])
            builder.set_scene_idxs([i])
            cake = builder.build(name=f"cake_blank-{i}")
            self.remove_from_state_dict_registry(cake)
            cakes.append(cake)
        self.cake = Actor.merge(cakes, name="cake_blank")
        self.add_to_state_dict_registry(self.cake)
        self.belt.register_rider(self.cake, patch_radius=self._cake_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._topping_names: list[str] = [str(name) for name in names]
        self._topping_display: list[str] = [display_name(spec_by_id[n]) for n in self._topping_names]
        specs = [spec_by_id[n] for n in self._topping_names]

        toppings: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.50])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"cake_topping-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            toppings.append(obj)
        self.topping = Actor.merge(toppings, name="cake_topping")
        self.add_to_state_dict_registry(self.topping)
        self.belt.register_rider(
            self.topping,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._topping_patch_radius for s in specs],
                dtype=torch.float32,
                device=device,
            ),
        )
        self._topping_half_z = torch.as_tensor([s.bbox[2] for s in specs], dtype=torch.float32, device=device)
        self._topping_half_xy = torch.as_tensor(
            [max(s.bbox[0], s.bbox[1]) for s in specs], dtype=torch.float32, device=device
        )
        self._topping_grasp_half = torch.as_tensor(
            [(s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1])) / 2.0 for s in specs],
            dtype=torch.float32,
            device=device,
        )

    @property
    def episode_objects(self) -> list[str]:
        return list(self._topping_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"put the {name} inside the printed target zone on the cake on the conveyor belt"
            for name in self._topping_display
        ]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = self._topping_grasp_half[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_top_trackers(env_idx)

        cake_pos = self._sample_cake_position(env_idx, b)
        cake_q = self._sample_cake_orientation(b)
        self.cake.set_pose(Pose.create_from_pq(cake_pos, cake_q))
        cake_vel = torch.zeros((b, 3), device=device)
        cake_vel[:, 1] = self.belt.velocity[env_idx]
        self.cake.set_linear_velocity(cake_vel)
        self.cake.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        topping_q = torch.zeros((b, 4), device=device)
        topping_q[:, 0] = 1.0
        self.topping.set_pose(Pose.create_from_pq(tcp_pos, topping_q))
        self.topping.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.topping.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _sample_cake_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.cake_yaw:
            lo, hi = self.randomization.cake_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_cake_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._cake_radius - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._cake_body_half_z + 0.001
        return pos

    def _reset_top_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_top_sustain_counter"):
            self._top_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._topping_fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._top_sustain_counter[env_idx] = 0
        self._topping_fell[env_idx] = False

    def evaluate(self):
        agent = cast("Panda", self.agent)
        cake_pose = self.cake.pose
        cake_pos = cake_pose.p
        topping_pos = self.topping.pose.p
        belt_v = self.belt.velocity
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        local = self._in_cake_frame(topping_pos)
        zone_dist = torch.linalg.norm(local[:, :2] - self._zone_offset, dim=-1)
        in_zone = zone_dist <= (self._zone_radius_t - self._topping_half_xy)

        footprint_inner = (torch.linalg.norm(local[:, :2], dim=-1) - self._topping_half_xy).clamp(min=0.0)
        surface_z = self._crown_surface_local_z(footprint_inner)
        resting = torch.abs((local[:, 2] - self._topping_half_z) - surface_z) <= self._rest_band

        finger_width = self.gripper_aperture()
        released = (finger_width >= self._released_finger_width) & (
            torch.linalg.norm(agent.tcp.pose.p - topping_pos, dim=-1) > self._released_distance_threshold
        )

        topping_v = self.topping.linear_velocity
        cake_v = self.cake.linear_velocity
        comoving = (
            (torch.abs(topping_v[:, 1] - belt_v) <= self._comove_tol_y)
            & (torch.abs(topping_v[:, 0]) <= self._comove_tol_x)
            & (torch.linalg.norm(topping_v - cake_v, dim=-1) < self._pair_comove_tol)
        )

        self._topping_fell = self._topping_fell | (topping_pos[:, 2] < deck_z + self._cake_height)

        cake_up = torch.zeros((self.num_envs, 3), device=self.device)
        cake_up[:, 2] = 1.0
        tilt_cos = ((cake_pose * Pose.create_from_pq(cake_up)).p - cake_pos)[:, 2]
        cake_on_belt = (
            (torch.abs(cake_pos[:, 0] - ox) <= self._cake_x_tol)
            & (torch.abs(cake_v[:, 1] - belt_v) <= self._cake_comove_tol)
            & (tilt_cos >= self._cake_tilt_cos)
            & (torch.abs(cake_pos[:, 1] - oy) < self._cake_y_limit)
        )

        topped = in_zone & resting & released & comoving & (~self._topping_fell) & cake_on_belt

        self._top_sustain_counter = self._advance_sustain("cake_top_sustain", self._top_sustain_counter, topped)
        success = self._top_sustain_counter >= self._top_sustain_steps

        return {
            "success": success,
            "in_zone": in_zone,
            "resting_on_crown": resting,
            "released": released,
            "comoving": comoving,
            "not_rolled_off": ~self._topping_fell,
            "cake_on_belt": cake_on_belt,
            "cake_top_sustain": self._top_sustain_counter,
            "topping_to_zone": zone_dist,
        }

    def compute_dense_reward(self, obs, action, info):
        topping_pos = self.topping.pose.p
        zone_pos = self.zone_world_pos()

        xy_dist = torch.linalg.norm(topping_pos[:, :2] - zone_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        seated = (info["in_zone"] & info["resting_on_crown"] & info["not_rolled_off"]).float()
        return carry_reward + 2.0 * seated

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
