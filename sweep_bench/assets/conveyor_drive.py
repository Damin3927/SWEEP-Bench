from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs.actor import Actor


@dataclass
class _Rider:
    actor: Actor
    patch_radius: float | torch.Tensor
    m_eff: torch.Tensor
    weight: torch.Tensor
    inertia: torch.Tensor
    integral: torch.Tensor


class BeltDrive:
    def __init__(
        self,
        scene: ManiSkillScene,
        belt_actor: Actor,
        *,
        deck_top_z: float,
        static_friction: float,
        kinetic_friction: float,
        axis: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._scene = scene
        self._belt_actor = belt_actor
        self._deck_top_z = deck_top_z
        self._static_friction = static_friction
        self._kinetic_friction = kinetic_friction
        if abs(math.hypot(*axis) - 1.0) > 1e-6:
            raise ValueError(f"axis must be a unit vector, got {axis}")
        self._axis = torch.tensor(axis, device=scene.device, dtype=torch.float32)
        self._pending: list[tuple[Actor, float | torch.Tensor]] = []
        self._riders: list[_Rider] | None = None
        self.velocity = torch.zeros(scene.num_envs, device=scene.device, dtype=torch.float32)
        self.velocity_np = np.zeros(scene.num_envs, dtype=np.float32)

    @property
    def direction_sign(self) -> torch.Tensor:
        return torch.where(self.velocity >= 0.0, 1.0, -1.0)

    @property
    def primary_rider(self) -> Actor:
        riders = self.riders
        if not riders:
            raise RuntimeError("no riders are registered on this belt, so it has no primary rider")
        return riders[0]

    @property
    def riders(self) -> list[Actor]:
        if self._riders:
            return [rider.actor for rider in self._riders]
        return [actor for actor, _ in self._pending]

    def register_rider(self, actor: Actor, *, patch_radius: float | torch.Tensor) -> None:
        if self._riders is not None:
            raise RuntimeError("register_rider must be called before the first reset()")
        if isinstance(patch_radius, torch.Tensor) and patch_radius.shape != (self._scene.num_envs,):
            raise ValueError(
                f"per-env patch_radius must have shape ({self._scene.num_envs},), got {patch_radius.shape}"
            )
        self._pending.append((actor, patch_radius))

    def reset(self, env_idx: torch.Tensor, velocity: torch.Tensor) -> None:
        if self._riders is None:
            self._riders = [self._make_rider(actor, radius) for actor, radius in self._pending]
        self.velocity[env_idx] = velocity
        self.velocity_np = self.velocity.cpu().numpy()
        for rider in self._riders:
            rider.integral[env_idx] = 0.0

    def _make_rider(self, actor: Actor, patch_radius: float | torch.Tensor) -> _Rider:
        device = self._scene.device
        mass = actor.mass.to(device)
        if isinstance(patch_radius, torch.Tensor):
            patch_radius = patch_radius.to(device=device, dtype=torch.float32)
        return _Rider(
            actor=actor,
            patch_radius=patch_radius,
            m_eff=0.5 * mass,
            weight=9.81 * mass,
            inertia=torch.as_tensor(
                np.array([body.inertia for body in actor._bodies], dtype=np.float32).mean(axis=1),
                device=device,
            ),
            integral=torch.zeros(self._scene.num_envs, device=device, dtype=torch.float32),
        )

    def step(self) -> None:
        self._apply(self.compute_writes())

    def compute_writes(self) -> list[tuple[Actor, torch.Tensor, torch.Tensor]]:
        if self._riders is None:
            return []
        if self._scene.gpu_sim_enabled:
            self._scene.px.gpu_fetch_rigid_dynamic_data()
        dt = 1.0 / self._scene.sim_config.sim_freq
        return [(rider.actor, *self._rider_force_torque(rider, dt)) for rider in self._riders]

    def _rider_force_torque(self, rider: _Rider, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs, device = self._scene.num_envs, self._scene.device
        actor = rider.actor

        contact_force = self._scene.get_pairwise_contact_forces(actor, self._belt_actor)
        normal = torch.minimum(contact_force[:, 2].clamp(min=0.0), 5.0 * rider.weight)

        lever = torch.zeros((num_envs, 3), device=device)
        lever[:, 2] = -(actor.pose.p[:, 2] - self._deck_top_z).clamp(0.01, 0.06)

        v_contact = actor.linear_velocity + torch.linalg.cross(actor.angular_velocity, lever)
        slip = v_contact[:, :2] - self.velocity.unsqueeze(-1) * self._axis

        cone = self._static_friction * normal
        f_stick = -(rider.m_eff / dt).unsqueeze(-1) * slip
        f_stick = f_stick + rider.integral.unsqueeze(-1) * self._axis
        slip_along = slip @ self._axis
        sticking = (torch.linalg.norm(f_stick, dim=-1) <= cone).unsqueeze(-1).to(slip.dtype)

        slip_dir = slip / torch.linalg.norm(slip, dim=-1, keepdim=True).clamp(min=1e-8)
        f_kinetic = -(self._kinetic_friction * normal).unsqueeze(-1) * slip_dir

        ki = 200.0
        i_cap = self._kinetic_friction * normal
        rider.integral = torch.clamp((rider.integral - ki * slip_along * dt) * sticking.squeeze(-1), -i_cap, i_cap)

        force = torch.zeros((num_envs, 3), device=device)
        force[:, :2] = sticking * f_stick + (1.0 - sticking) * f_kinetic
        torque_drive = torch.linalg.cross(lever, force)

        tau_db = -(rider.inertia / dt).unsqueeze(-1) * actor.angular_velocity
        resting = (normal <= 1.5 * rider.weight).unsqueeze(-1).to(slip.dtype)
        tau_cap = (rider.patch_radius * torch.minimum(normal, 1.5 * rider.weight)).unsqueeze(-1)
        tau_patch = torch.clamp(tau_db - resting * torque_drive, -tau_cap, tau_cap)

        return force, torque_drive + tau_patch

    def _apply(self, writes: list[tuple[Actor, torch.Tensor, torch.Tensor]]) -> None:
        apply_drive_writes(self._scene, writes)


def apply_drive_writes(scene: ManiSkillScene, writes: list[tuple[Actor, torch.Tensor, torch.Tensor]]) -> None:
    if scene.gpu_sim_enabled:
        px = scene.px
        force_buf = px.cuda_rigid_body_force.torch()
        torque_buf = px.cuda_rigid_body_torque.torch()
        for actor, force, torque in writes:
            force_buf[actor._body_data_index, :3] = force
            torque_buf[actor._body_data_index, :3] = torque
        px.gpu_apply_rigid_dynamic_force()
        px.gpu_apply_rigid_dynamic_torque()
    else:
        for actor, force, torque in writes:
            actor._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), torque.squeeze(0).cpu().numpy())
