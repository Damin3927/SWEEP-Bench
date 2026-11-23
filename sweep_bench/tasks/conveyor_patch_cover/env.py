from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_patch_cover.assets import build_patch_board
from sweep_bench.tasks.conveyor_patch_cover.config import RandomizationConfig
from sweep_bench.tasks.conveyor_patch_cover.scripted import ScriptedPatchCoverMixin
from sweep_bench.tasks.conveyor_patch_cover.variants import (
    PATCH_COVER_TILE_IDS,
    PATCH_COVER_TILE_SPECS,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec

_BOARD_COLOR = (0.34, 0.38, 0.46, 1.0)
_BOARD_PAD_COLOR = (0.18, 0.18, 0.20, 1.0)

_PATCH_COLORS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("red", (0.85, 0.12, 0.12, 1.0)),
    ("blue", (0.13, 0.30, 0.75, 1.0)),
)

_TILE_IN_HAND_Q = torch.tensor([0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0], dtype=torch.float32)


@bench_task("ConveyorPatchCover-v0")
@register_env("ConveyorPatchCover-v0", max_episode_steps=300)
class ConveyorPatchCoverEnv(ScriptedPatchCoverMixin, ConveyorWorkspaceEnv):
    """Cover the NAMED coloured patch on a moving board, leave the other showing."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(PATCH_COVER_TILE_IDS, extra=PATCH_COVER_TILE_SPECS)

    _board_half_x: float = 0.055
    _board_half_y: float = 0.115
    _board_half_z: float = 0.005
    _board_pad_half_height: float = 0.0015
    _board_pad_static_friction: float = 0.9
    _board_pad_dynamic_friction: float = 0.8
    _board_patch_radius: float = 0.050

    _patch_half_x: float = 0.010
    _patch_half_y: float = 0.010
    _patch_sep: float = 0.12

    _coverage_margin: float = 0.004
    _exclusion_margin: float = 0.004
    _decoy_latch_z: float = 0.015
    _flat_cos: float = math.cos(math.radians(10.0))
    _flat_contact_tol: float = 0.004
    _released_finger_margin: float = 0.01
    _released_distance_threshold: float = 0.08
    _comove_tol: float = 0.05
    _board_x_tol: float = 0.04
    _board_track_tol: float = 0.08
    _board_level_cos: float = math.cos(math.radians(10.0))
    _board_span_tol: float = 0.85
    _cover_sustain_steps: int = 10

    _default_named_index: int = 0

    default_prompt: str = "cover the red patch on the board on the conveyor belt with the tile"

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
            per_load_axis = (objects is not None and len(objects) > 1) or randomization.patch_swap
            reconfiguration_freq = 1 if (num_envs == 1 and per_load_axis) else 0
        super().__init__(
            *args,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            randomization=randomization,
            objects=objects,
            **kwargs,
        )

    def _to_world(self, pose: Pose, local: torch.Tensor) -> torch.Tensor:
        return (pose * Pose.create_from_pq(local)).p

    def _patch_local(self, index: torch.Tensor) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 1] = torch.gather(self._patch_y, 1, index.unsqueeze(-1)).squeeze(-1)
        local[:, 2] = self._board_half_z
        return local

    def named_patch_world_pos(self) -> torch.Tensor:
        return self._to_world(self.board.pose, self._patch_local(self._named_index))

    def _assert_tolerance_budget(self, specs: list[ObjectSpec]) -> None:
        for spec in specs:
            half_x, half_y = spec.bbox[0], spec.bbox[1]
            cover_x = half_x - self._patch_half_x - self._coverage_margin
            cover_y = half_y - self._patch_half_y - self._coverage_margin
            exclusion = self._patch_sep - self._patch_half_y - half_y - self._exclusion_margin
            if cover_x <= 0.0 or cover_y <= 0.0:
                raise ValueError(
                    f"tile {spec.id!r} (half {half_x} x {half_y}) cannot cover a "
                    f"{self._patch_half_x} x {self._patch_half_y} patch with a "
                    f"{self._coverage_margin} m margin"
                )
            if exclusion < 2.0 * max(cover_x, cover_y):
                raise ValueError(
                    f"tile {spec.id!r}: exclusion_slack {exclusion:.4f} m < 2 * coverage_slack "
                    f"{max(cover_x, cover_y):.4f} m -- the decoy term would become the binding "
                    f"constraint (see ConveyorPatchCoverEnv's tolerance budget)"
                )
            if 0.5 * self._patch_sep + half_y > self._board_half_y or half_x > self._board_half_x:
                raise ValueError(
                    f"tile {spec.id!r} (half {half_x} x {half_y}) overhangs the "
                    f"{self._board_half_x} x {self._board_half_y} board when centred on a patch"
                )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        if self.randomization.patch_swap:
            swapped = np.asarray(self._batched_episode_rng.choice(np.array([0, 1]))).reshape(-1).astype(bool)
        else:
            swapped = np.zeros(self.num_envs, dtype=bool)
        half_sep = 0.5 * self._patch_sep
        patch_y = np.stack([np.where(swapped, -half_sep, half_sep), np.where(swapped, half_sep, -half_sep)], axis=1)
        self._patch_y = torch.as_tensor(patch_y, dtype=torch.float32, device=self.device)

        boards: list[Actor] = []
        for i in range(self.num_envs):
            builder = build_patch_board(
                self.scene,
                half_x=self._board_half_x,
                half_y=self._board_half_y,
                half_z=self._board_half_z,
                pad_half_height=self._board_pad_half_height,
                pad_static_friction=self._board_pad_static_friction,
                pad_dynamic_friction=self._board_pad_dynamic_friction,
                color=_BOARD_COLOR,
                pad_color=_BOARD_PAD_COLOR,
                patch_half_x=self._patch_half_x,
                patch_half_y=self._patch_half_y,
                patches=[(float(patch_y[i, c]), rgba) for c, (_name, rgba) in enumerate(_PATCH_COLORS)],
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._board_half_z + 0.001])
            builder.set_scene_idxs([i])
            board = builder.build(name=f"patch_cover_board-{i}")
            self.remove_from_state_dict_registry(board)
            boards.append(board)
        self.board = Actor.merge(boards, name="patch_cover_board")
        self.add_to_state_dict_registry(self.board)
        self.belt.register_rider(self.board, patch_radius=self._board_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._tile_names: list[str] = [str(name) for name in names]
        specs = [spec_by_id[name] for name in self._tile_names]
        self._tile_display: list[str] = [display_name(spec) for spec in specs]
        self._assert_tolerance_budget(specs)

        tiles: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.50])
            builder.set_scene_idxs([i])
            tile = builder.build(name=f"patch_cover_tile-{spec.id}-{i}")
            self.remove_from_state_dict_registry(tile)
            tiles.append(tile)
        self.tile = Actor.merge(tiles, name="patch_cover_tile")
        self.add_to_state_dict_registry(self.tile)
        self.belt.register_rider(
            self.tile,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else min(s.bbox[0], s.bbox[1]) for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        def _col(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        self._tile_half_x = _col([s.bbox[0] for s in specs])
        self._tile_half_y = _col([s.bbox[1] for s in specs])
        self._tile_half_z = _col([s.bbox[2] for s in specs])
        self._tile_grasp_width = _col([s.grasp_width if s.grasp_width is not None else 2.0 * s.bbox[0] for s in specs])

    @property
    def episode_objects(self) -> list[str]:
        return list(self._tile_names)

    @property
    def episode_prompts(self) -> list[str]:
        named = self._named_index.cpu().tolist()
        return [
            f"cover the {_PATCH_COLORS[int(idx)][0]} patch on the board with the {tile}, "
            f"leave the {_PATCH_COLORS[1 - int(idx)][0]} one showing"
            for idx, tile in zip(named, self._tile_display, strict=True)
        ]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = 0.5 * self._tile_grasp_width[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_cover_trackers(env_idx)
        self._named_index[env_idx] = self._sample_named_index(b)

        board_pos = self._sample_board_position(env_idx, b)
        board_q = torch.zeros((b, 4), device=device)
        board_q[:, 0] = 1.0
        self.board.set_pose(Pose.create_from_pq(board_pos, board_q))
        board_vel = torch.zeros((b, 3), device=device)
        board_vel[:, 1] = self.belt.velocity[env_idx]
        self.board.set_linear_velocity(board_vel)
        self.board.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        tcp_pos = tcp_pose.p[env_idx]
        tile_q = quaternion_multiply(tcp_pose.q[env_idx], _TILE_IN_HAND_Q.to(device).expand(b, 4))
        self.tile.set_pose(Pose.create_from_pq(tcp_pos, tile_q))
        self.tile.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.tile.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _sample_named_index(self, b: int) -> torch.Tensor:
        if self.randomization.named_patch:
            picks = self._episode_rng.choice(np.array([0, 1]), size=b)
            return torch.as_tensor(picks, dtype=torch.long, device=self.device)
        return torch.full((b,), self._default_named_index, dtype=torch.long, device=self.device)

    def _sample_board_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._board_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._board_half_z + 0.001
        return pos

    def _reset_cover_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_cover_sustain_counter"):
            self._cover_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._named_index = torch.full(
                (self.num_envs,), self._default_named_index, dtype=torch.long, device=self.device
            )
            self._decoy_shadowed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cover_sustain_counter[env_idx] = 0
        self._decoy_shadowed[env_idx] = False

    def _rect_separation(
        self,
        tile_xy: torch.Tensor,
        rot: torch.Tensor,
        patch_xy: torch.Tensor,
    ) -> torch.Tensor:
        u = rot[:, :2, 0]
        v = rot[:, :2, 1]
        delta = tile_xy - patch_xy
        e1 = torch.zeros_like(u)
        e1[:, 0] = 1.0
        e2 = torch.zeros_like(u)
        e2[:, 1] = 1.0
        u_hat = u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp(min=1e-9)
        v_hat = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-9)

        best = None
        for axis in (e1, e2, u_hat, v_hat):
            centre_gap = torch.abs((delta * axis).sum(dim=-1))
            tile_extent = self._tile_half_x * torch.abs((u * axis).sum(dim=-1)) + self._tile_half_y * torch.abs(
                (v * axis).sum(dim=-1)
            )
            patch_extent = self._patch_half_x * torch.abs(axis[:, 0]) + self._patch_half_y * torch.abs(axis[:, 1])
            gap = centre_gap - tile_extent - patch_extent
            best = gap if best is None else torch.maximum(best, gap)
        assert best is not None
        return best

    def evaluate(self):
        agent = cast("Panda", self.agent)
        board_pose = self.board.pose
        board_pos = board_pose.p
        tile_pos = self.tile.pose.p
        belt_v = self.belt.velocity

        rel = board_pose.inv() * self.tile.pose
        rel_p = rel.p
        rot = quaternion_to_matrix(rel.q)

        named_local = self._patch_local(self._named_index)
        decoy_local = self._patch_local(1 - self._named_index)

        offsets = torch.tensor(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        corners = named_local.unsqueeze(1).repeat(1, 4, 1)
        corners[:, :, 0] = corners[:, :, 0] + offsets[:, 0] * self._patch_half_x
        corners[:, :, 1] = corners[:, :, 1] + offsets[:, 1] * self._patch_half_y
        local_corners = torch.matmul(corners - rel_p.unsqueeze(1), rot)
        corner_reach_x = local_corners[:, :, 0].abs().amax(dim=1)
        corner_reach_y = local_corners[:, :, 1].abs().amax(dim=1)
        covered = (corner_reach_x <= self._tile_half_x - self._coverage_margin) & (
            corner_reach_y <= self._tile_half_y - self._coverage_margin
        )

        decoy_separation = self._rect_separation(rel_p[:, :2], rot, decoy_local[:, :2])
        tile_underside = rel_p[:, 2] - self._tile_half_z * rot[:, 2, 2]
        at_board_level = tile_underside - self._board_half_z < self._decoy_latch_z
        self._decoy_shadowed = self._decoy_shadowed | (at_board_level & (decoy_separation < self._exclusion_margin))
        decoy_clear = ~self._decoy_shadowed

        upright = rot[:, 2, 2] >= self._flat_cos
        on_board = torch.abs(tile_underside - self._board_half_z) <= self._flat_contact_tol
        lying_flat = upright & on_board

        finger_width = self.gripper_aperture()
        released = (finger_width > self._tile_grasp_width + self._released_finger_margin) & (
            torch.linalg.norm(agent.tcp.pose.p - tile_pos, dim=-1) > self._released_distance_threshold
        )

        riding = torch.linalg.norm(self.tile.linear_velocity - self.board.linear_velocity, dim=-1) < self._comove_tol

        ox, oy, _oz = self.workspace_offset
        board_up_z = quaternion_to_matrix(board_pose.q)[:, 2, 2]
        board_on_belt = (
            (torch.abs(board_pos[:, 0] - ox) <= self._board_x_tol)
            & (torch.abs(self.board.linear_velocity[:, 1] - belt_v) <= self._board_track_tol)
            & (board_up_z >= self._board_level_cos)
            & (torch.abs(board_pos[:, 1] - oy) <= self._board_span_tol)
        )

        covering = covered & decoy_clear & lying_flat & released & riding & board_on_belt

        self._cover_sustain_counter = self._advance_sustain(
            "patch_cover_sustain", self._cover_sustain_counter, covering
        )
        success = self._cover_sustain_counter >= self._cover_sustain_steps

        tile_yaw_deg = torch.rad2deg(torch.atan2(rot[:, 1, 0], rot[:, 0, 0])).abs()

        return {
            "success": success,
            "covered": covered,
            "decoy_clear": decoy_clear,
            "lying_flat": lying_flat,
            "released": released,
            "riding": riding,
            "board_on_belt": board_on_belt,
            "decoy_separation": decoy_separation,
            "corner_reach_x": corner_reach_x,
            "corner_reach_y": corner_reach_y,
            "tile_yaw_deg": tile_yaw_deg,
            "cover_sustain": self._cover_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        tile_pos = self.tile.pose.p
        patch_pos = self.named_patch_world_pos()

        xy_dist = torch.linalg.norm(tile_pos[:, :2] - patch_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        laid = (info["covered"] & info["lying_flat"] & info["decoy_clear"]).float()
        return carry_reward + 2.0 * laid

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
