from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_sheet_edge_pick.assets import (
    build_framed_sheet,
    build_pane,
    build_sheet_rack,
)
from sweep_bench.tasks.conveyor_sheet_edge_pick.config import RandomizationConfig
from sweep_bench.tasks.conveyor_sheet_edge_pick.scripted import ScriptedSheetEdgePickMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_PALM_LINK = {
    "panda": "panda_hand",
    "panda_wristcam": "panda_hand",
}


@bench_task("ConveyorSheetEdgePick-v0")
@register_env("ConveyorSheetEdgePick-v0", max_episode_steps=300)
class ConveyorSheetEdgePickEnv(ScriptedSheetEdgePickMixin, ConveyorWorkspaceEnv):
    """Pinch a framed sheet off its moving rack by the frame's lip, never touching the face."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _rack_half_x: float = 0.070
    _rack_half_y: float = 0.080
    _rack_plate_half_thickness: float = 0.006
    _rib_half_x: float = 0.045
    _rib_half_y: float = 0.006
    _rib_y: float = 0.065
    _rib_height: float = 0.110
    _rack_density: float = 1500.0
    _rack_patch_radius: float = 0.045

    _sheet_half_x: float = 0.065
    _sheet_half_y: float = 0.075
    _rail_width_x: float = 0.042
    _rail_width_y: float = 0.020
    _rail_thickness: float = 0.012
    _ledge_width: float = 0.006
    _ledge_thickness: float = 0.006
    _sheet_density: float = 800.0
    _pane_clearance: float = 0.0015
    _pane_half_thickness: float = 0.003
    _pane_density: float = 1000.0

    _entry_aperture: float = 0.036
    _approach_tilt_deg: float = 20.0

    _PINCH_REST_QPOS_PLUS: ClassVar[tuple[float, ...]] = (
        -0.4450,
        -1.0196,
        0.7202,
        -2.7532,
        -1.9224,
        2.8478,
        1.7752,
    )
    _PINCH_REST_QPOS_MINUS: ClassVar[tuple[float, ...]] = (
        0.4427,
        -1.0180,
        -0.7188,
        -2.7536,
        1.9220,
        2.8456,
        -0.2027,
    )

    _grip_gap_tol: float = 0.006
    _hold_dist: float = 0.075
    _lift_clearance: float = 0.050
    _lift_min_corner_clearance: float = 0.005
    _pane_seat_tol: float = 0.008
    _pane_rabbet_rise: float = _rail_thickness - _ledge_thickness
    _pane_float_budget: int = 5
    _pane_seat_max_tilt_cos: float = math.cos(math.radians(30.0))
    _pane_aperture_penetration_tol: float = 0.0015
    _contact_force_threshold: float = 1e-3
    _rack_max_tilt_cos: float = math.cos(math.radians(10.0))
    _rack_z_tol: float = 0.006
    _static_carrier_travel_tol: float = 0.015
    _rack_speed_tol_base: float = 0.06
    _rack_speed_tol_per_speed: float = 0.20
    _carrier_stall_limit: int = 10

    _carrier_travel_frac: float = 0.5
    _pick_sustain_steps: int = 5

    default_prompt: str = "lift the framed sheet off its rack by the frame edge without touching the sheet face"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _rack_origin_z(self) -> float:
        return 2.0 * self._rack_plate_half_thickness

    @property
    def _rack_rest_z(self) -> float:
        return self.workspace_offset[2] + self._conveyor_deck_height + self._rack_origin_z

    @property
    def _rib_top_local_z(self) -> float:
        return self._rib_height

    @property
    def _pane_half_x(self) -> float:
        return self._sheet_half_x - self._rail_width_x - self._pane_clearance

    @property
    def _pane_half_y(self) -> float:
        return self._sheet_half_y - self._rail_width_y - self._pane_clearance

    @property
    def _pane_seat_local_z(self) -> float:
        return self._ledge_thickness + self._pane_half_thickness

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        rack_builder = build_sheet_rack(
            self.scene,
            plate_half_x=self._rack_half_x,
            plate_half_y=self._rack_half_y,
            plate_half_thickness=self._rack_plate_half_thickness,
            rib_half_x=self._rib_half_x,
            rib_half_y=self._rib_half_y,
            rib_height=self._rib_height,
            rib_y=self._rib_y,
            density=self._rack_density,
        )
        rack_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._rack_origin_z])
        self.rack = rack_builder.build(name="sheet_rack")
        self.belt.register_rider(self.rack, patch_radius=self._rack_patch_radius)

        sheet_builder = build_framed_sheet(
            self.scene,
            half_x=self._sheet_half_x,
            half_y=self._sheet_half_y,
            rail_width_x=self._rail_width_x,
            rail_width_y=self._rail_width_y,
            rail_thickness=self._rail_thickness,
            ledge_width=self._ledge_width,
            ledge_thickness=self._ledge_thickness,
            density=self._sheet_density,
        )
        sheet_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._rack_origin_z + self._rib_top_local_z])
        self.sheet = sheet_builder.build(name="framed_sheet")

        pane_builder = build_pane(
            self.scene,
            half_x=self._pane_half_x,
            half_y=self._pane_half_y,
            half_thickness=self._pane_half_thickness,
            density=self._pane_density,
        )
        pane_builder.initial_pose = sapien.Pose(
            p=[ox, oy, deck_z + self._rack_origin_z + self._rib_top_local_z + self._pane_seat_local_z]
        )
        self.pane = pane_builder.build(name="sheet_pane")

        seat = torch.zeros((self.num_envs, 3), device=self.device)
        seat[:, 2] = self._pane_seat_local_z
        self._pane_seat_offset_pose = Pose.create_from_pq(p=seat)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.sheet, self.pane]

    @property
    def _primary_rider(self) -> Actor:
        return self.sheet

    def _reset_rest_qpos(self, env_idx: torch.Tensor) -> None:
        if self.robot_uids not in ("panda", "panda_wristcam"):
            raise ValueError(
                f"{type(self).__name__} has a pinch-frame rest pose only for the panda "
                f"(got robot_uids={self.robot_uids!r}); see SUPPORTED_ROBOTS."
            )
        b = len(env_idx)
        sign = self.belt.direction_sign[env_idx].cpu().numpy().reshape(-1, 1)
        arm_rest = np.where(
            sign > 0.0,
            np.asarray(self._PINCH_REST_QPOS_PLUS, dtype=np.float64),
            np.asarray(self._PINCH_REST_QPOS_MINUS, dtype=np.float64),
        )
        gripper_rest = self._rest_gripper_qpos(env_idx, b)
        qpos_np = self._episode_rng.normal(0, 0.02, (b, arm_rest.shape[1] + gripper_rest.shape[1]))
        qpos_np[:, : arm_rest.shape[1]] += arm_rest
        qpos_np[:, arm_rest.shape[1] :] = gripper_rest
        self.agent.reset(torch.as_tensor(qpos_np, dtype=torch.float32, device=self.device))

    def _initialize_episode_draws(self, env_idx: torch.Tensor, options: dict) -> None:
        env_idx = env_idx.to(self.belt.velocity.device)
        super()._initialize_episode_draws(env_idx, options)
        self._reset_rest_qpos(env_idx)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        rack_pos = self._sample_rack_position(env_idx, b)
        rack_q = torch.zeros((b, 4), device=device)
        rack_q[:, 0] = 1.0
        self.rack.set_pose(Pose.create_from_pq(rack_pos, rack_q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.rack.set_linear_velocity(vel)
        self.rack.set_angular_velocity(torch.zeros((b, 3), device=device))

        seat_dx, seat_dy = self._sample_seat_offset(b)
        sheet_pos = rack_pos.clone()
        sheet_pos[:, 0] = sheet_pos[:, 0] + seat_dx
        sheet_pos[:, 1] = sheet_pos[:, 1] + seat_dy
        sheet_pos[:, 2] = sheet_pos[:, 2] + self._rib_top_local_z + 0.001
        self.sheet.set_pose(Pose.create_from_pq(sheet_pos, rack_q))
        self.sheet.set_linear_velocity(vel)
        self.sheet.set_angular_velocity(torch.zeros((b, 3), device=device))

        pane_pos = sheet_pos.clone()
        pane_pos[:, 2] = pane_pos[:, 2] + self._pane_seat_local_z
        self.pane.set_pose(Pose.create_from_pq(pane_pos, rack_q))
        self.pane.set_linear_velocity(vel)
        self.pane.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_pick_trackers(env_idx)

    def _sample_seat_offset(self, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        if not self.randomization.sheet_seat_offset:
            zeros = torch.zeros((b,), device=device)
            return zeros, zeros.clone()
        lo, hi = self.randomization.seat_range
        dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        dy = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(dx, device=device), torch.as_tensor(dy, device=device)

    def _sample_rack_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._rack_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rack_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = oz + self._conveyor_deck_height + self._rack_origin_z
        return pos

    def _rack_rotation(self) -> torch.Tensor:
        return quaternion_to_matrix(self.rack.pose.q)

    def _rack_half_extent(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot = self._rack_rotation()
        half = torch.tensor([self._rack_half_x, self._rack_half_y], device=self.device)
        span = rot[:, :2, :2].abs() @ half
        return span[:, 0], span[:, 1]

    def _rack_support_centre(self) -> torch.Tensor:
        return self._rack_rotation()[:, :2, 2] * (-2.0 * self._rack_plate_half_thickness)

    def _pane_aperture_slack(self) -> tuple[torch.Tensor, torch.Tensor]:
        rel = quaternion_to_matrix(self.sheet.pose.q).transpose(1, 2) @ quaternion_to_matrix(self.pane.pose.q)
        half = torch.tensor([self._pane_half_x, self._pane_half_y], device=self.device)
        span = rel[:, :2, :2].abs() @ half
        span = span + rel[:, :2, 2].abs() * self._pane_half_thickness
        aperture = torch.tensor(
            [self._sheet_half_x - self._rail_width_x, self._sheet_half_y - self._rail_width_y],
            device=self.device,
        )
        slack = aperture - span - torch.abs(self._pane_seat_error()[:, :2])
        return slack[:, 0], slack[:, 1]

    def _rack_local(self, world_points: torch.Tensor) -> torch.Tensor:
        rel = world_points - self.rack.pose.p
        return torch.bmm(self._rack_rotation().transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)

    def _pane_seat_error(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.sheet.pose.q)
        rel = self.pane.pose.p - self.sheet.pose.p
        local = torch.bmm(rot.transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)
        return local - self._pane_seat_offset_pose.p

    def _pane_overhang_is_pressed(self) -> torch.Tensor:
        rel = quaternion_to_matrix(self.sheet.pose.q).transpose(1, 2) @ quaternion_to_matrix(self.pane.pose.q)
        half3 = torch.tensor([self._pane_half_x, self._pane_half_y, self._pane_half_thickness], device=self.device)
        error = self._pane_seat_error()
        aperture = torch.tensor(
            [self._sheet_half_x - self._rail_width_x, self._sheet_half_y - self._rail_width_y],
            device=self.device,
        )
        centre_z = self._pane_seat_local_z + error[:, 2]

        box = (
            torch.tensor(
                [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)], device=self.device
            )
            * half3
        )
        reach = torch.einsum("bij,cj->bic", rel[:, :2, :], box)
        height = centre_z[:, None] + torch.einsum("bj,cj->bc", rel[:, 2, :], box)

        pressed = []
        for axis in (0, 1):
            unpressed_side = torch.zeros_like(centre_z, dtype=torch.bool)
            for side in (1.0, -1.0):
                extent = side * (error[:, axis, None] + reach[:, axis])
                deepest = extent.amax(dim=-1, keepdim=True)
                at_deepest = extent >= deepest - 1e-6
                corner_z = torch.where(at_deepest, height, torch.full_like(height, float("inf"))).amin(dim=-1)
                sticks_out = deepest.squeeze(-1) > aperture[axis]
                unpressed_side = unpressed_side | (sticks_out & (corner_z >= self._rail_thickness))
            pressed.append(~unpressed_side)
        return torch.stack(pressed, dim=-1)

    def _pane_seat_tilt_cos(self) -> torch.Tensor:
        rel = quaternion_to_matrix(self.sheet.pose.q).transpose(1, 2) @ quaternion_to_matrix(self.pane.pose.q)
        return rel[:, 2, 2]

    def _sheet_lift_height(self) -> torch.Tensor:
        return self._rack_local(self.sheet.pose.p)[:, 2] - self._rib_top_local_z

    def _sheet_lowest_corner_clearance(self) -> torch.Tensor:
        rack_rot = self._rack_rotation().transpose(1, 2)
        rel = rack_rot @ quaternion_to_matrix(self.sheet.pose.q)
        half = torch.tensor([self._sheet_half_x, self._sheet_half_y], device=self.device)
        drop = (rel[:, 2, :2].abs() * half).sum(dim=-1)
        drop = drop + (-rel[:, 2, 2] * self._rail_thickness).clamp(min=0.0)
        frame = self._sheet_lift_height() - drop

        pane_rel = rack_rot @ quaternion_to_matrix(self.pane.pose.q)
        pane_half = torch.tensor([self._pane_half_x, self._pane_half_y, self._pane_half_thickness], device=self.device)
        pane_drop = (pane_rel[:, 2, :].abs() * pane_half).sum(dim=-1)
        pane = self._rack_local(self.pane.pose.p)[:, 2] - pane_drop - self._rib_top_local_z
        return torch.minimum(frame, pane)

    def _reset_pick_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_pick_sustain_counter"):
            self._pick_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._pane_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._rack_spawn_xy = torch.zeros((self.num_envs, 2), device=self.device)
            self._rack_stall_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._rack_stalled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._pane_float_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._pane_floated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._pre_pinch_riding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._was_pinched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._caught_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pick_sustain_counter[env_idx] = 0
        self._pane_touched[env_idx] = False
        self._rack_stall_steps[env_idx] = 0
        self._rack_stalled[env_idx] = False
        self._pane_float_steps[env_idx] = 0
        self._pane_floated[env_idx] = False
        self._pre_pinch_riding[env_idx] = True
        self._was_pinched[env_idx] = False
        self._caught_moving[env_idx] = True
        self._rack_spawn_xy[env_idx] = self.rack.pose.p[env_idx, :2]

    def _after_simulation_step(self) -> None:
        super()._after_simulation_step()
        self._latch_pane_contact()

    def _latch_pane_contact(self) -> None:
        if not hasattr(self, "_pane_touched"):
            return
        agent = cast("Panda", self.agent)
        palm_name = _PALM_LINK.get(self.robot_uids)
        if palm_name is None:
            raise ValueError(
                f"{type(self).__name__} has no gripper palm link for robot_uids={self.robot_uids!r}; "
                f"known: {sorted(_PALM_LINK)}"
            )
        for link in (agent.finger1_link, agent.finger2_link, agent.robot.links_map[palm_name]):
            force = torch.linalg.norm(self.scene.get_pairwise_contact_forces(link, self.pane), dim=-1)
            self._pane_touched = self._pane_touched | (force > self._contact_force_threshold)

    def evaluate(self):
        ox, oy, _oz = self.workspace_offset
        agent = cast("Panda", self.agent)
        rack_pos = self.rack.pose.p
        sheet_pos = self.sheet.pose.p
        tcp_pos = agent.tcp.pose.p

        aperture = self.gripper_aperture()
        pinched = torch.abs(aperture - self._rail_thickness) < self._grip_gap_tol
        at_hand = torch.linalg.norm(sheet_pos - tcp_pos, dim=-1) < self._hold_dist
        sheet_held = agent.is_grasping(self.sheet) & pinched & at_hand

        lift_height = self._sheet_lift_height()
        corner_clearance = self._sheet_lowest_corner_clearance()
        sheet_lifted = (lift_height >= self._lift_clearance) & (corner_clearance >= self._lift_min_corner_clearance)

        face_untouched = ~self._pane_touched

        pane_level = self._pane_seat_tilt_cos() > self._pane_seat_max_tilt_cos
        slack_x, slack_y = self._pane_aperture_slack()
        pressed = self._pane_overhang_is_pressed()
        allowed = torch.where(
            pressed,
            torch.full_like(pressed, -self._pane_aperture_penetration_tol, dtype=slack_x.dtype),
            torch.zeros_like(slack_x).unsqueeze(-1).expand_as(pressed),
        )
        pane_in_aperture = (slack_x >= allowed[:, 0]) & (slack_y >= allowed[:, 1])
        pane_in_rabbet = self._pane_seat_error()[:, 2] <= self._pane_rabbet_rise
        self._pane_float_steps = (
            self._pane_float_steps + (~pane_in_rabbet).long() * self._eval_advanced("sheet_edge_pane_float").long()
        )
        self._pane_floated = self._pane_floated | (self._pane_float_steps >= self._pane_float_budget)
        pane_seated = (
            (torch.abs(self._pane_seat_error()) < self._pane_seat_tol).all(dim=-1)
            & pane_level
            & pane_in_aperture
            & ~self._pane_floated
        )

        half_x, half_y = self._rack_half_extent()
        support = rack_pos[:, :2] + self._rack_support_centre()
        on_belt = (torch.abs(support[:, 0] - ox) <= self._conveyor_half_width - half_x) & (
            torch.abs(support[:, 1] - oy) <= self._conveyor_half_length - half_y
        )
        at_belt_height = torch.abs(rack_pos[:, 2] - self._rack_rest_z) <= self._rack_z_tol
        belt_v = self.belt.velocity
        rack_v_y = self.rack.linear_velocity[:, 1]
        speed_tol = self._rack_speed_tol_base + self._rack_speed_tol_per_speed * tolerance_speed(belt_v)
        tracking = torch.abs(rack_v_y - belt_v) <= speed_tol
        travelling = self.belt_carried(self.rack, self._carrier_travel_frac)
        rack_travel = torch.linalg.norm(rack_pos[:, :2] - self._rack_spawn_xy, dim=-1)
        stationary = (torch.abs(belt_v) > 0.0) | (rack_travel <= self._static_carrier_travel_tol)
        upright = self._rack_rotation()[:, 2, 2] > self._rack_max_tilt_cos
        rack_riding = on_belt & at_belt_height & tracking & travelling & stationary & upright

        capturing = ~(sheet_held & sheet_lifted)
        advanced = self._eval_advanced("sheet_edge_stall")
        self._rack_stall_steps = self._rack_stall_steps + (capturing & ~travelling).long() * advanced.long()
        self._rack_stalled = self._rack_stalled | (self._rack_stall_steps >= self._carrier_stall_limit)
        rack_never_stalled = ~self._rack_stalled

        pre_pinch = advanced & ~sheet_held
        self._pre_pinch_riding = torch.where(pre_pinch, travelling & at_belt_height, self._pre_pinch_riding)
        first_pinch = sheet_held & ~self._was_pinched
        new_attempt = first_pinch & ~(sheet_lifted)
        regrasp = first_pinch & (sheet_lifted)
        self._caught_moving = torch.where(new_attempt, self._pre_pinch_riding, self._caught_moving)
        self._caught_moving = torch.where(regrasp, self._caught_moving & self._pre_pinch_riding, self._caught_moving)
        self._was_pinched = torch.where(advanced, sheet_held, self._was_pinched)

        holding = (
            sheet_held
            & sheet_lifted
            & face_untouched
            & pane_seated
            & rack_riding
            & rack_never_stalled
            & self._caught_moving
        )
        self._pick_sustain_counter = self._advance_sustain(
            "sheet_edge_pick_sustain", self._pick_sustain_counter, holding
        )
        success = self._pick_sustain_counter >= self._pick_sustain_steps

        return {
            "success": success,
            "sheet_held": sheet_held,
            "sheet_pinched": pinched,
            "sheet_at_hand": at_hand,
            "sheet_lifted": sheet_lifted,
            "sheet_lift_height": lift_height,
            "sheet_corner_clearance": corner_clearance,
            "face_untouched": face_untouched,
            "pane_seated": pane_seated,
            "pane_level": pane_level,
            "pane_in_aperture": pane_in_aperture,
            "rack_on_belt": on_belt,
            "rack_at_belt_height": at_belt_height,
            "rack_tracking": tracking,
            "rack_stationary": stationary,
            "rack_travelling": travelling,
            "rack_upright": upright,
            "rack_riding": rack_riding,
            "rack_never_stalled": rack_never_stalled,
            "caught_moving": self._caught_moving,
            "pane_in_rabbet": pane_in_rabbet,
            "pick_sustain": self._pick_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        sheet_pos = self.sheet.pose.p

        lip_local = torch.zeros_like(sheet_pos)
        lip_local[:, 0] = -self._sheet_half_x
        lip_local[:, 2] = 0.5 * self._rail_thickness
        lip = (self.sheet.pose * Pose.create_from_pq(lip_local)).p
        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - lip, dim=-1))

        lift_reward = (self._sheet_lift_height() / self._lift_clearance).clamp(0.0, 1.0)
        earned = (
            info["sheet_held"]
            & info["face_untouched"]
            & info["pane_seated"]
            & info["rack_never_stalled"]
            & info["caught_moving"]
        ).float()

        return reach_reward + 2.0 * lift_reward * earned

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
