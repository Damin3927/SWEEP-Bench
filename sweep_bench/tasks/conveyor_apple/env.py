from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_apple.config import RandomizationConfig
from sweep_bench.tasks.conveyor_apple.scripted import ScriptedGraspMixin
from sweep_bench.tasks.conveyor_apple.variants import APPLE_GRASP_OBJECT_IDS
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorAppleGrasp-v0")
@register_env("ConveyorAppleGrasp-v0", max_episode_steps=300)
class ConveyorAppleGraspEnv(ScriptedGraspMixin, ConveyorWorkspaceEnv):
    """Grasp an object moving on a conveyor belt (default: the YCB apple)."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(APPLE_GRASP_OBJECT_IDS)

    _patch_radius: float = 0.02

    default_prompt: str = "pick up the apple from the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._rider_names: list[str] = [str(name) for name in names]
        self._rider_display: list[str] = [display_name(spec_by_id[name]) for name in self._rider_names]
        self._rider_objs: list[Actor] = []
        patch_radii: list[float] = []
        for i, name in enumerate(self._rider_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy - 0.3, oz + self._conveyor_deck_height + 0.05])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            obj.angular_damping = 5.0
            self._rider_objs.append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._patch_radius)
        self.apple = Actor.merge(self._rider_objs, name="apple")
        self.add_to_state_dict_registry(self.apple)
        self.belt.register_rider(
            self.apple,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=self.device),
        )

        self.lift_height = oz + self._conveyor_deck_height + 0.15

    @property
    def episode_objects(self) -> list[str]:
        return list(self._rider_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"pick up the {name} from the conveyor belt" for name in self._rider_display]

    def _after_reconfigure(self, options: dict) -> None:
        verts: list[torch.Tensor] = []
        for obj in self._rider_objs:
            mesh = obj.get_first_collision_mesh(to_world_frame=False)
            assert mesh is not None, f"rider actor {obj.name} has no collision mesh"
            verts.append(torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device))
        v_max = max(v.shape[0] for v in verts)
        self._rider_hull_verts = torch.stack([torch.cat([v, v[-1:].expand(v_max - v.shape[0], 3)]) for v in verts])

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        rcfg = self.randomization

        if rcfg.apple_orientation:
            u1 = self._episode_rng.uniform(0.0, 1.0, size=b)
            u2 = self._episode_rng.uniform(0.0, 2 * np.pi, size=b)
            u3 = self._episode_rng.uniform(0.0, 2 * np.pi, size=b)
            sigma1 = np.sqrt(1.0 - u1)
            sigma2 = np.sqrt(u1)
            qw = np.cos(u3) * sigma2
            qx = np.sin(u2) * sigma1
            qy = np.cos(u2) * sigma1
            qz = np.sin(u3) * sigma2
            apple_q = torch.as_tensor(
                np.stack([qw, qx, qy, qz], axis=-1),
                device=device,
                dtype=torch.float32,
            )
        else:
            apple_q = torch.zeros((b, 4), device=device, dtype=torch.float32)
            apple_q[:, 0] = 1.0

        rot = quaternion_to_matrix(apple_q)
        verts = self._rider_hull_verts[env_idx]
        lowest = torch.einsum("bj,bvj->bv", rot[:, 2, :], verts).amin(dim=-1)
        spawn_z = self._conveyor_deck_height - lowest + 0.001
        apple_pos = self._sample_apple_position(env_idx, b, spawn_z)

        self.apple.set_pose(Pose.create_from_pq(apple_pos, apple_q))
        apple_vel = torch.zeros((b, 3), device=device)
        apple_vel[:, 1] = self.belt.velocity[env_idx]
        self.apple.set_linear_velocity(apple_vel)
        self.apple.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_grasp_trackers(env_idx)

    def _sample_apple_position(self, env_idx: torch.Tensor, b: int, spawn_z: torch.Tensor) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - 0.06
        apple_pos = torch.zeros((b, 3), device=device)
        apple_pos[:, 0] = ox + 0.0
        apple_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        apple_pos[:, 2] = oz + spawn_z
        return apple_pos

    _grasp_distance_threshold: float = 0.06
    _apple_static_speed_threshold: float = 0.3
    _grasp_sustain_steps: int = 5

    def _reset_grasp_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_grasp_sustain_counter"):
            self._grasp_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_sustain_counter[env_idx] = 0

    def evaluate(self):
        agent = cast("Panda", self.agent)
        apple_pos = self.apple.pose.p
        tcp_pos = agent.tcp.pose.p
        apple_speed = torch.linalg.norm(self.apple.linear_velocity, dim=-1)

        is_grasped = agent.is_grasping(self.apple) & (
            torch.linalg.norm(tcp_pos - apple_pos, dim=-1) < self._grasp_distance_threshold
        )
        is_lifted = apple_pos[:, 2] > self.lift_height
        apple_static = apple_speed < self._apple_static_speed_threshold

        holding = is_grasped & is_lifted & apple_static
        self._grasp_sustain_counter = self._advance_sustain("apple_grasp", self._grasp_sustain_counter, holding)
        success = self._grasp_sustain_counter >= self._grasp_sustain_steps

        return {
            "success": success,
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
            "apple_static": apple_static,
            "grasp_sustain": self._grasp_sustain_counter,
            "apple_z": apple_pos[:, 2],
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        apple_pos = self.apple.pose.p

        tcp_to_apple = torch.linalg.norm(tcp_pos - apple_pos, dim=-1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_apple)

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        lift_reward = torch.clamp((apple_pos[:, 2] - deck_z) / 0.15, 0, 1)

        return reaching_reward + 2.0 * lift_reward * info["is_lifted"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
