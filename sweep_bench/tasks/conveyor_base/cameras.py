from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import sapien
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

if TYPE_CHECKING:
    from mani_skill.agents.base_agent import BaseAgent
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs.pose import Pose

    from sweep_bench.tasks.conveyor_base.config import SceneStyle

Offset = tuple[float, float, float]

SIDE_PLUS_Y_EYE: Offset = (0.6, -0.4, 0.72)
SIDE_PLUS_Y_TARGET: Offset = (0.0, 0.0, 0.27)
SIDE_FOV = 1.0
HEAD_EYE: Offset = (0.48, 0.0, 1.40)
HEAD_TARGET: Offset = (-0.39, 0.0, 0.25)
HEAD_FOV = 0.85


def mirror_y(offset: Offset) -> Offset:
    x, y, z = offset
    return (x, -y + 0.0, z)


EXTERIOR_RIG: tuple[tuple[str, Offset, Offset, float], ...] = (
    ("head_camera", HEAD_EYE, HEAD_TARGET, HEAD_FOV),
    ("side_camera_plus_y", SIDE_PLUS_Y_EYE, SIDE_PLUS_Y_TARGET, SIDE_FOV),
    ("side_camera_minus_y", mirror_y(SIDE_PLUS_Y_EYE), mirror_y(SIDE_PLUS_Y_TARGET), SIDE_FOV),
)


def shifted_look_at(workspace_offset: Offset, eye, target):
    ox, oy, oz = workspace_offset
    eye = [eye[0] + ox, eye[1] + oy, eye[2] + oz]
    target = [target[0] + ox, target[1] + oy, target[2] + oz]
    return sapien_utils.look_at(eye=eye, target=target)


def human_render_camera_config(
    workspace_offset: Offset,
    shader_pack: str,
    *,
    eye: Offset = (0.85, 0.0, 0.8),
    target: Offset = (0.0, 0.0, 0.35),
) -> CameraConfig:
    pose = shifted_look_at(workspace_offset, list(eye), list(target))
    return CameraConfig(
        "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100, shader_pack=shader_pack
    )


def viewer_camera_config(shader_pack: str) -> CameraConfig:
    return CameraConfig(
        uid="viewer",
        pose=cast("Pose", sapien.Pose([0, 0, 1])),
        width=1920,
        height=1080,
        shader_pack=shader_pack,
        near=0.0,
        far=1000,
        fov=math.pi / 2,
    )


def sensor_camera_configs(
    workspace_offset: Offset,
    shader_pack: str,
    agent: BaseAgent,
    *,
    eye_jitters: tuple[Offset, ...] = ((0.0, 0.0, 0.0),) * len(EXTERIOR_RIG),
) -> list[CameraConfig]:
    if len(eye_jitters) != len(EXTERIOR_RIG):
        raise ValueError(f"expected {len(EXTERIOR_RIG)} exterior eye jitters, got {len(eye_jitters)}")

    configs = [
        CameraConfig(
            uid,
            pose=shifted_look_at(
                workspace_offset,
                [eye[0] + jx, eye[1] + jy, eye[2] + jz],
                list(target),
            ),
            width=224,
            height=224,
            fov=fov,
            near=0.01,
            far=100,
            shader_pack=shader_pack,
        )
        for (uid, eye, target, fov), (jx, jy, jz) in zip(EXTERIOR_RIG, eye_jitters, strict=True)
    ]

    links = agent.robot.links_map
    if "camera_link" in links:
        wrist_pose = cast("Pose", sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
        configs.append(
            CameraConfig(
                "wrist_camera",
                pose=wrist_pose,
                width=224,
                height=224,
                fov=math.pi / 2,
                near=0.01,
                far=100,
                mount=links["camera_link"],
                shader_pack=shader_pack,
            )
        )

    return configs


def build_lighting(
    scene: ManiSkillScene,
    scene_style: SceneStyle,
    workspace_offset: Offset,
    *,
    intensity_scale: float = 1.0,
    color_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    key_dir_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    ox, oy, oz = workspace_offset
    cr, cg, cb = color_scale
    scale = [intensity_scale * cr, intensity_scale * cg, intensity_scale * cb]
    key_dir = [1.0 + key_dir_offset[0], 1.0 + key_dir_offset[1], -2.0 + key_dir_offset[2]]

    if scene_style == "replicacad":
        scene.add_directional_light(
            key_dir,
            [0.8 * scale[0], 0.78 * scale[1], 0.73 * scale[2]],
            shadow=True,
            position=[ox, oy, oz + 2.0],
            shadow_scale=1.5,
            shadow_map_size=2048,
        )
        return

    scene.set_ambient_light([0.3 * scale[0], 0.3 * scale[1], 0.3 * scale[2]])
    scene.add_directional_light(
        key_dir,
        [1.0 * scale[0], 0.95 * scale[1], 0.9 * scale[2]],
        shadow=True,
        position=[ox, oy, oz + 2.0],
        shadow_scale=1.5,
        shadow_map_size=2048,
    )
    scene.add_directional_light([-1.0, -0.5, -1.0], [0.3 * scale[0], 0.3 * scale[1], 0.35 * scale[2]])
