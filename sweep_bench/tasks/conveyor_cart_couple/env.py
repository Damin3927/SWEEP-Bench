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

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_cart_couple.assets import build_follower_cart, build_lead_cart
from sweep_bench.tasks.conveyor_cart_couple.config import RandomizationConfig
from sweep_bench.tasks.conveyor_cart_couple.scripted import ScriptedCartCoupleMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorCartCouple-v0")
@register_env("ConveyorCartCouple-v0", max_episode_steps=300)
class ConveyorCartCoupleEnv(ScriptedCartCoupleMixin, ConveyorWorkspaceEnv):
    """Hold the lead cart back until the follower catches up, then let the pair ride on."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    conveyor_speed: float = 0.10

    _cart_half_x: float = 0.045
    _cart_height: float = 0.030
    _lead_half_y: float = 0.0325
    _follower_half_y: float = 0.045
    _socket_depth: float = 0.045
    _channel_half_x: float = 0.022
    _channel_floor_z: float = 0.005
    _crest_z: float = 0.0085
    _crest_y: float = -0.0485
    _mouth_flare: float = 0.016
    _mouth_flare_run: float = 0.020
    _stub_half_x: float = 0.012
    _stub_half_y: float = 0.030
    _stub_half_z: float = 0.016
    _blade_half_x: float = 0.018
    _blade_reach: float = 0.045
    _blade_bottom_z: float = 0.0135
    _blade_top_z: float = 0.0235
    _toe_len: float = 0.012
    _toe_bottom_z: float = 0.0075

    _lead_density: float = 420.0
    _follower_density: float = 640.0
    _coupling_friction: float = 0.08

    _cart_patch_radius: float = 0.03

    _lead_start_y: float = 0.51
    _cart_gap_default: float = 0.30

    _engage_min_penetration: float = 0.012
    _engage_depth_slack: float = 0.004
    _engage_max_lateral: float = 0.006
    _engage_z_slack: float = 0.004
    _mated_max_gap: float = 0.008
    _hands_off_distance: float = 0.10
    _release_finger_margin: float = 0.010
    _riding_speed_floor: float = 0.06
    _riding_speed_frac: float = 0.15
    _belt_moving_threshold: float = 0.032
    _lane_half_width: float = 0.05
    _max_tilt_cos: float = math.cos(math.radians(10.0))
    _couple_sustain_steps: int = 8

    default_prompt: str = "hold the front cart until the following cart couples to it, then let the pair ride on"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        if reconfiguration_freq is None:
            scene_load_axes = randomization.socket_clearance or randomization.cart_mass
            reconfiguration_freq = 1 if (num_envs == 1 and scene_load_axes) else 0
        super().__init__(
            *args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, randomization=randomization, **kwargs
        )

    @property
    def _lead_origin_z(self) -> float:
        return self._cart_height + self._stub_half_z

    @property
    def _follower_origin_z(self) -> float:
        return self._cart_height * 0.5

    @property
    def _socket_mouth_y(self) -> float:
        return -(self._lead_half_y + self._socket_depth)

    @property
    def _mated_center_distance(self) -> float:
        return self._lead_half_y + self._socket_depth + self._follower_half_y

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device

        channel_half_xs = self._sample_channel_half_widths()
        lead_densities, follower_densities = self._sample_densities()

        lead_objs: list[Actor] = []
        follower_objs: list[Actor] = []
        for i in range(self.num_envs):
            lead_builder = build_lead_cart(
                self.scene,
                body_half_x=self._cart_half_x,
                body_half_y=self._lead_half_y,
                body_height=self._cart_height,
                socket_depth=self._socket_depth,
                channel_half_x=channel_half_xs[i],
                channel_floor_z=self._channel_floor_z,
                crest_z=self._crest_z,
                crest_y=self._crest_y,
                mouth_flare=self._mouth_flare,
                mouth_flare_run=self._mouth_flare_run,
                stub_half_x=self._stub_half_x,
                stub_half_y=self._stub_half_y,
                stub_half_z=self._stub_half_z,
                density=lead_densities[i],
                coupling_friction=self._coupling_friction,
            )
            lead_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._lead_origin_z])
            lead_builder.set_scene_idxs([i])
            lead_obj = lead_builder.build(name=f"lead_cart-{i}")
            self.remove_from_state_dict_registry(lead_obj)
            lead_objs.append(lead_obj)

            follower_builder = build_follower_cart(
                self.scene,
                body_half_x=self._cart_half_x,
                body_half_y=self._follower_half_y,
                body_height=self._cart_height,
                blade_half_x=self._blade_half_x,
                blade_reach=self._blade_reach,
                blade_bottom_z=self._blade_bottom_z,
                blade_top_z=self._blade_top_z,
                toe_len=self._toe_len,
                toe_bottom_z=self._toe_bottom_z,
                density=follower_densities[i],
                coupling_friction=self._coupling_friction,
            )
            follower_builder.initial_pose = sapien.Pose(p=[ox, oy - 0.3, deck_z + self._follower_origin_z])
            follower_builder.set_scene_idxs([i])
            follower_obj = follower_builder.build(name=f"follower_cart-{i}")
            self.remove_from_state_dict_registry(follower_obj)
            follower_objs.append(follower_obj)

        self.lead = Actor.merge(lead_objs, name="lead_cart")
        self.follower = Actor.merge(follower_objs, name="follower_cart")
        self.add_to_state_dict_registry(self.lead)
        self.add_to_state_dict_registry(self.follower)
        self.belt.register_rider(self.lead, patch_radius=self._cart_patch_radius)
        self.belt.register_rider(self.follower, patch_radius=self._cart_patch_radius)

        self._channel_half_x_per_env = torch.as_tensor(channel_half_xs, dtype=torch.float32, device=device)

        tip = torch.zeros((self.num_envs, 3), device=device)
        tip[:, 1] = self._follower_half_y + self._blade_reach
        tip[:, 2] = 0.5 * (self._toe_bottom_z + self._blade_bottom_z) - self._follower_origin_z
        self._tongue_tip_offset = Pose.create_from_pq(p=tip)

        mouth = torch.zeros((self.num_envs, 3), device=device)
        mouth[:, 1] = self._socket_mouth_y
        self._socket_mouth_offset = Pose.create_from_pq(p=mouth)

        front = torch.zeros((self.num_envs, 3), device=device)
        front[:, 1] = self._follower_half_y
        self._follower_front_offset = Pose.create_from_pq(p=front)

    def _sample_channel_half_widths(self) -> list[float]:
        if not self.randomization.socket_clearance:
            return [self._channel_half_x] * self.num_envs
        lo, hi = self.randomization.channel_half_x_range
        return [float(v) for v in self._episode_rng.uniform(lo, hi, size=self.num_envs)]

    def _sample_densities(self) -> tuple[list[float], list[float]]:
        if not self.randomization.cart_mass:
            return (
                [self._lead_density] * self.num_envs,
                [self._follower_density] * self.num_envs,
            )
        lead = self._episode_rng.uniform(*self.randomization.lead_density_range, size=self.num_envs)
        follower = self._episode_rng.uniform(*self.randomization.follower_density_range, size=self.num_envs)
        return [float(v) for v in lead], [float(v) for v in follower]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        sign = self.belt.direction_sign[env_idx]

        gaps = self._sample_cart_gaps(b)

        lane_x, lead_y, follower_y = self._sample_cart_xy(env_idx, b, gaps)

        lead_pos = torch.zeros((b, 3), device=device)
        lead_pos[:, 0] = lane_x
        lead_pos[:, 1] = lead_y
        lead_pos[:, 2] = deck_z + self._lead_origin_z + 0.001

        follower_pos = torch.zeros((b, 3), device=device)
        follower_pos[:, 0] = lane_x
        follower_pos[:, 1] = follower_y
        follower_pos[:, 2] = deck_z + self._follower_origin_z + 0.001

        base_yaw = torch.where(sign > 0, torch.zeros_like(sign), torch.full_like(sign, math.pi))
        lead_yaw = base_yaw + self._sample_lead_yaw_jitter(b)
        self.lead.set_pose(Pose.create_from_pq(lead_pos, self._yaw_quat(lead_yaw)))
        self.follower.set_pose(Pose.create_from_pq(follower_pos, self._yaw_quat(base_yaw)))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)
        for cart in (self.lead, self.follower):
            cart.set_linear_velocity(vel)
            cart.set_angular_velocity(zero)

        self._reset_couple_trackers(env_idx)

    def _sample_cart_xy(
        self, env_idx: torch.Tensor, b: int, gaps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ox, oy, _oz = self.workspace_offset
        sign = self.belt.direction_sign[env_idx]
        return (
            torch.full((b,), ox, device=self.device),
            oy - self._lead_start_y * sign,
            oy - (self._lead_start_y + gaps) * sign,
        )

    def _sample_cart_gaps(self, b: int) -> torch.Tensor:
        if not self.randomization.cart_gap:
            return torch.full((b,), self._cart_gap_default, device=self.device)
        lo, hi = self.randomization.cart_gap_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _sample_lead_yaw_jitter(self, b: int) -> torch.Tensor:
        if not self.randomization.lead_yaw:
            return torch.zeros((b,), device=self.device)
        lo, hi = self.randomization.lead_yaw_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _yaw_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        q = torch.zeros((yaw.shape[0], 4), device=self.device)
        q[:, 0] = torch.cos(yaw * 0.5)
        q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _reset_couple_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_couple_counter"):
            self._couple_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._couple_counter[env_idx] = 0

    def _cart_on_lane(self, pose: Pose, deck_z: float) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        pos = pose.p
        up = quaternion_to_matrix(pose.q)[:, 2, 2]
        return (
            (torch.abs(pos[:, 0] - ox) <= self._lane_half_width)
            & (torch.abs(pos[:, 1] - oy) < self._conveyor_half_length)
            & (pos[:, 2] > deck_z)
            & (up >= self._max_tilt_cos)
        )

    def _cart_slip(self, cart: Actor, belt_v: torch.Tensor) -> torch.Tensor:
        return torch.abs(cart.linear_velocity[:, 1] - belt_v)

    def _cart_settled(self, cart: Actor) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        sign = self.belt.direction_sign
        lead_pose = self.lead.pose
        follower_pose = self.follower.pose

        tip_local = (lead_pose.inv() * (follower_pose * self._tongue_tip_offset)).p
        penetration = tip_local[:, 1] - self._socket_mouth_y
        engaged = (
            (penetration >= self._engage_min_penetration)
            & (penetration <= self._socket_depth + self._engage_depth_slack)
            & (torch.abs(tip_local[:, 0]) <= self._engage_max_lateral)
            & (tip_local[:, 2] >= self._channel_floor_z - self._lead_origin_z - self._engage_z_slack)
            & (tip_local[:, 2] <= self._cart_height - self._lead_origin_z)
        )

        mouth_world = (lead_pose * self._socket_mouth_offset).p
        front_world = (follower_pose * self._follower_front_offset).p
        mate_gap = (mouth_world[:, 1] - front_world[:, 1]) * sign
        mated = mate_gap <= self._mated_max_gap

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        finger_width = self.gripper_aperture()
        released = finger_width >= 2.0 * self._stub_half_x + self._release_finger_margin
        withdrawn = (torch.linalg.norm(tcp_pos - lead_pose.p, dim=-1) >= self._hands_off_distance) & (
            torch.linalg.norm(tcp_pos - follower_pose.p, dim=-1) >= self._hands_off_distance
        )
        hands_off = released & withdrawn

        belt_v = self.belt.velocity
        riding_tol = torch.clamp(tolerance_speed(belt_v) * self._riding_speed_frac, min=self._riding_speed_floor)
        lead_slip = self._cart_slip(self.lead, belt_v)
        follower_slip = self._cart_slip(self.follower, belt_v)
        both_riding = (
            (lead_slip <= riding_tol)
            & (follower_slip <= riding_tol)
            & self._cart_settled(self.lead)
            & self._cart_settled(self.follower)
        )

        belt_running = torch.abs(belt_v) >= self._belt_moving_threshold

        on_lane = self._cart_on_lane(lead_pose, deck_z) & self._cart_on_lane(follower_pose, deck_z)

        holding = engaged & mated & hands_off & both_riding & belt_running & on_lane
        self._couple_counter = self._advance_sustain("cart_couple", self._couple_counter, holding)
        success = self._couple_counter >= self._couple_sustain_steps

        return {
            "success": success,
            "engaged": engaged,
            "mated": mated,
            "hands_off": hands_off,
            "released": released,
            "withdrawn": withdrawn,
            "both_riding": both_riding,
            "belt_running": belt_running,
            "on_lane": on_lane,
            "couple_sustain": self._couple_counter,
            "mate_gap": mate_gap,
            "tongue_penetration": penetration,
            "tongue_lateral": torch.abs(tip_local[:, 0]),
            "lead_slip": lead_slip,
            "follower_slip": follower_slip,
        }

    def compute_dense_reward(self, obs, action, info):
        sign = self.belt.direction_sign
        couple_gap = (self.lead.pose.p[:, 1] - self.follower.pose.p[:, 1]) * sign - self._mated_center_distance
        close_reward = 1 - torch.tanh(8 * couple_gap.abs())
        coupled_reward = (info["engaged"] & info["mated"]).float()
        couple_reward = (info["couple_sustain"].float() / self._couple_sustain_steps).clamp(0.0, 1.0)
        return close_reward + coupled_reward + 2.0 * couple_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
